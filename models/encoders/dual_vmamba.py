import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from pathlib import Path

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
import math
import time
from engine.logger import get_logger
from models.bwm_runtime import resolve_bwm_enabled, validate_bwm_mode, is_main_process
from models.encoders.vmamba import (
    Backbone_VSSM,
    CrossMambaFusionBlock,
    ConcatMambaFusionBlock,
    CrossModalStateBlock,
    WRCSIBlock,
    RCSFBlock,
)

logger = get_logger()


def _resolve_pretrained(pretrained):
    """Resolve DWR initialization weights independently of the working directory."""
    if pretrained is None:
        return None
    path = Path(pretrained).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.is_file():
        raise FileNotFoundError(
            f"VMamba initialization weights not found: {path}. "
            "Download the matching ImageNet checkpoint as described in README.md. "
            "Evaluation with a trained DWR checkpoint uses pretrained=None."
        )
    return str(path)


class RGBXTransformer(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()

        self.ape = ape

        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        self.cross_mamba = nn.ModuleList(
            #CrossMambaFusionBlock(
            CrossModalStateBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )

        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)

                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x C x H x W
        """
        B = x_rgb.shape[0]
        outs_fused = []

        outs_rgb = self.vssm(x_rgb) # B x C x H x W
        outs_x = self.vssm(x_e) # B x C x H x W

        for i in range(4):
            if self.ape:
                # this has been discarded
                out_rgb = self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
                out_x = self.absolute_pos_embed_x[i].to(outs_x[i].device) + outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]
            cross_rgb, cross_x = self.cross_mamba[i](
                out_rgb.permute(0, 2, 3, 1).contiguous(),
                out_x.permute(0, 2, 3, 1).contiguous(),
            )
            x_fuse = self.channel_attn_mamba[i](
                cross_rgb,
                cross_x,
            ).permute(0, 3, 1, 2).contiguous()
            outs_fused.append(x_fuse)
        return outs_fused

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out

class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )




# =====================================================================
# DWR-Mamba main network
#
# Hierarchical direction-strength ranges with WRCSI and RCSF.
# =====================================================================


class DWRBackbone(RGBXTransformer):
    def __init__(
        self,
        num_classes=1000,
        norm_layer=nn.LayerNorm,
        depths=[2, 2, 27, 2],
        dims=96,
        pretrained=None,
        mlp_ratio=4.0,
        downsample_version='v1',
        ape=False,
        img_size=[480, 640],
        patch_size=4,
        drop_path_rate=0.2,
        router_hidden=32,
        router_init_std=0.02,
        topk_ratio=0.05,
        power_eps=1e-4,
        detach_router_stats=True,
        ccs_reduction: int = 16,
        ccs_min_hidden: int = 16,
        ccs_scale: float = 0.1,
        detach_ccs_stats: bool = True,
        use_ri: bool = True,
        use_wi: bool = True,
        use_bwm=None,
        use_rcsf: bool = True,
        use_rcsf_feat: bool = True,
        use_rcsf_diff: bool = True,
        use_rcsf_coact: bool = True,
        bwm_mode: str = "full",
        **kwargs,
    ):
        use_bwm = resolve_bwm_enabled(use_bwm, kwargs.pop("use_hbwr", None))
        bwm_mode = validate_bwm_mode(bwm_mode)
        super().__init__(
            num_classes=num_classes,
            norm_layer=norm_layer,
            depths=depths,
            dims=dims,
            pretrained=pretrained,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            ape=ape,
            img_size=img_size,
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

        self.use_ri = bool(use_ri)
        self.use_wi = bool(use_wi)
        self.use_bwm = use_bwm
        self.bwm_mode = bwm_mode
        self.use_rcsf = bool(use_rcsf)
        self.use_rcsf_feat = bool(use_rcsf_feat)
        self.use_rcsf_diff = bool(use_rcsf_diff)
        self.use_rcsf_coact = bool(use_rcsf_coact)

        self.use_wrcsi = self.use_wi or self.use_ri
        self.use_state_interaction = self.use_wrcsi or self.use_bwm

        if isinstance(dims, int):
            stage_dims = [int(dims * (2 ** i)) for i in range(4)]
        else:
            stage_dims = dims

        # -------------------------------------------------------------
        # v3 change 1:
        # low-level conservative, high-level semantic-enhanced.
        # -------------------------------------------------------------
        stage_s_ranges = [
            (0.75, 1.25),  # stage0: detail-preserving
            (0.75, 1.25),  # stage1: conservative
            (0.50, 1.50),  # stage2: semantic modulation
            (0.40, 1.60),  # stage3: stronger high-level direction
        ]

        if self.use_state_interaction:
            self.cross_mamba = nn.ModuleList(
                WRCSIBlock(
                    hidden_dim=stage_dims[i],
                    mlp_ratio=0.0,
                    d_state=4,
                    router_hidden=router_hidden,
                    router_init_std=router_init_std,
                    s_min=stage_s_ranges[i][0],
                    s_max=stage_s_ranges[i][1],
                    topk_ratio=topk_ratio,
                    power_eps=power_eps,
                    detach_router_stats=detach_router_stats,
                    use_ri=use_ri,
                    use_wi=use_wi,
                    use_bwm=use_bwm,
                    bwm_mode=bwm_mode,
                    bwm_stage=i + 1,
                )
                for i in range(4)
            )
        else:
            self.cross_mamba = nn.ModuleList()

        # -------------------------------------------------------------
        # v3 change 2:
        # replace vanilla ConcatMambaFusionBlock with CCS version.
        # -------------------------------------------------------------
        if self.use_rcsf:
            self.channel_attn_mamba = nn.ModuleList(
                RCSFBlock(
                    hidden_dim=stage_dims[i],
                    mlp_ratio=0.0,
                    d_state=4,
                    ccs_reduction=ccs_reduction,
                    ccs_min_hidden=ccs_min_hidden,
                    ccs_scale=ccs_scale,
                    detach_ccs_stats=detach_ccs_stats,
                    use_rcsf=True,
                    use_rcsf_feat=use_rcsf_feat,
                    use_rcsf_diff=use_rcsf_diff,
                    use_rcsf_coact=use_rcsf_coact,
                )
                for i in range(4)
            )
        else:
            self.channel_attn_mamba = nn.ModuleList()

        if is_main_process():
            logger.info(
                "[BWM build] backbone=%s WI=%s RI=%s BWM=%s RCSF=%s mode=%s",
                type(self).__name__, self.use_wi, self.use_ri, self.use_bwm,
                self.use_rcsf, self.bwm_mode,
            )
            for block in self.cross_mamba:
                core = block.op.CMA_ssm
                router = core.direction_router
                logger.info(
                    "[BWM build] stage=%s enabled=%s core_mode=%s router_mode=%s "
                    "s_range=(%s, %s)",
                    core.bwm_stage, core.use_bwm, core.bwm_mode, router.bwm_mode,
                    router.s_min, router.s_max,
                )
            if not self.use_state_interaction:
                logger.info("[BWM build] state interaction bypassed; no BWM branch executes.")
            if not self.use_rcsf:
                logger.info("[BWM build] RCSF disabled; fusion=cross_rgb + cross_x (unchanged).")

    @property
    def use_hbwr(self):
        """Legacy switch alias; the canonical switch is use_bwm."""
        return self.use_bwm

    @use_hbwr.setter
    def use_hbwr(self, enabled):
        self.use_bwm = bool(enabled)

    def forward_features(self, x_rgb, x_e):
        outs_fused = []

        outs_rgb = self.vssm(x_rgb)
        outs_x = self.vssm(x_e)

        for i in range(4):
            if self.ape:
                out_rgb = self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
                out_x = self.absolute_pos_embed_x[i].to(outs_x[i].device) + outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]

            out_rgb_bhwc = out_rgb.permute(0, 2, 3, 1).contiguous()
            out_x_bhwc = out_x.permute(0, 2, 3, 1).contiguous()

            if self.use_state_interaction:
                cross_rgb, cross_x = self.cross_mamba[i](
                    out_rgb_bhwc,
                    out_x_bhwc,
                )
            else:
                cross_rgb, cross_x = out_rgb_bhwc, out_x_bhwc

            if self.use_rcsf:
                x_fuse = self.channel_attn_mamba[i](cross_rgb, cross_x)
            else:
                x_fuse = cross_rgb + cross_x

            outs_fused.append(x_fuse.permute(0, 3, 1, 2).contiguous())

        return outs_fused


class DWRTiny(DWRBackbone):
    def __init__(self, fuse_cfg=None,
                 pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
                 **kwargs):
        super().__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained=_resolve_pretrained(pretrained),
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
            detach_router_stats=True,
            detach_ccs_stats=True,
            **kwargs,
        )


class DWRSmall(DWRBackbone):
    def __init__(self, fuse_cfg=None,
                 pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
                 **kwargs):
        super().__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained=_resolve_pretrained(pretrained),
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
            detach_router_stats=True,
            detach_ccs_stats=True,
            **kwargs,
        )


class DWRBase(DWRBackbone):
    def __init__(self, fuse_cfg=None,
                 pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
                 **kwargs):
        super().__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained=_resolve_pretrained(pretrained),
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6,
            detach_router_stats=True,
            detach_ccs_stats=True,
            **kwargs,
        )


# =====================================================================
# DCR Main Network
# Only replace CroMB with CrossMambaFusionBlock_DCR.
# Keep original ConMB: ConcatMambaFusionBlock.
# =====================================================================


class RGBXTransformer_DCR(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        norm_layer=nn.LayerNorm,
        depths=[2, 2, 27, 2],
        dims=96,
        pretrained=None,
        mlp_ratio=4.0,
        downsample_version='v1',
        ape=False,
        img_size=[480, 640],
        patch_size=4,
        drop_path_rate=0.2,
        c_mix_scale=0.5,
        **kwargs,
    ):
        super().__init__()

        self.ape = ape

        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        # New DCR-CroMB
        self.cross_mamba = nn.ModuleList(
            CrossMambaFusionBlock_DCR(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
                c_mix_scale=c_mix_scale,
            )
            for i in range(4)
        )

        # Keep original Sigma ConMB unchanged
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            )
            for i in range(4)
        )

        if self.ape:
            self.patches_resolution = [
                img_size[0] // patch_size,
                img_size[1] // patch_size,
            ]

            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []

            for i_layer in range(len(depths)):
                input_resolution = (
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer),
                )
                dim = int(dims * (2 ** i_layer))

                absolute_pos_embed = nn.Parameter(
                    torch.zeros(1, dim, input_resolution[0], input_resolution[1])
                )
                trunc_normal_(absolute_pos_embed, std=.02)

                absolute_pos_embed_x = nn.Parameter(
                    torch.zeros(1, dim, input_resolution[0], input_resolution[1])
                )
                trunc_normal_(absolute_pos_embed_x, std=.02)

                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B, C, H, W
        x_e:   B, C, H, W
        """
        outs_fused = []

        outs_rgb = self.vssm(x_rgb)
        outs_x = self.vssm(x_e)

        for i in range(4):
            if self.ape:
                out_rgb = self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
                out_x = self.absolute_pos_embed_x[i].to(outs_x[i].device) + outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]

            out_rgb_bhwc = out_rgb.permute(0, 2, 3, 1).contiguous()
            out_x_bhwc = out_x.permute(0, 2, 3, 1).contiguous()

            # DCR-CroMB
            cross_rgb, cross_x = self.cross_mamba[i](
                out_rgb_bhwc,
                out_x_bhwc,
            )

            # Original ConMB
            x_fuse = self.channel_attn_mamba[i](
                cross_rgb,
                cross_x,
            ).permute(0, 3, 1, 2).contiguous()

            outs_fused.append(x_fuse)

        return outs_fused

    def forward(self, x_rgb, x_e):
        return self.forward_features(x_rgb, x_e)


class vssm_tiny_dcr(RGBXTransformer_DCR):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny_dcr, self).__init__(
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
            **kwargs,
        )


class vssm_small_dcr(RGBXTransformer_DCR):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small_dcr, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
            **kwargs,
        )


class vssm_base_dcr(RGBXTransformer_DCR):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base_dcr, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6,
            **kwargs,
        )
