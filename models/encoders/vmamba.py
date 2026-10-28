import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

from models.bwm_runtime import (
    resolve_bwm_enabled,
    validate_bwm_mode,
    emit_bwm_forward_evidence,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from functools import partial
from einops import rearrange, repeat
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except:
    pass

# cross selective scan ===============================
if True:
    import selective_scan_cuda_core as selective_scan_cuda

    class SelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

    class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)

    class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            return y

        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            return xs, None, None

    class CrossScan_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor, x_e: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W
            B, C, H, W = x_rgb.shape
            ctx.shape = (B, C, H, W)
            xs_fuse = x_rgb.new_empty((B, 2, C, 2 * H * W))
            xs_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, H, W = ctx.shape
            L = 2 * H * W
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W
            # get B, d, H*W
            return ys[:, :, 0:H*W].view(B, -1, H, W), ys[:, :, H*W:2*H*W].view(B, -1, H, W)

    class CrossMerge_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, L = ys.shape
            # ctx.shape = (H, W)
            # ys = ys.view(B, K, D, -1)
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W, broadcast
            # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
            return ys[:, :, 0:L//2], ys[:, :, L//2:L]

        @staticmethod
        def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            # H, W = ctx.shape
            B, C, L = x1.shape
            xs = x1.new_empty((B, 2, C, 2*L))
            xs[:, 0] = torch.cat([x1, x2], dim=2)
            xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            xs = xs.view(B, 2, C, 2*L)
            return xs, None, None

    def cross_selective_scan(
        x: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = CrossScan.apply(x)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)

        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)

        return y


    def selective_scan_1d(
        x: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        A_logs = A_logs[: A_logs.shape[0] // 4]
        Ds = Ds[: Ds.shape[0] // 4]
        B, D, H, W = x.shape
        D, N = A_logs.shape
        # get 1st of dt_projs_weight
        x_proj_weight = x_proj_weight[0].unsqueeze(0)
        x_proj_bias = x_proj_bias[0].unsqueeze(0) if x_proj_bias is not None else None
        dt_projs_weight = dt_projs_weight[0].unsqueeze(0)
        dt_projs_bias = dt_projs_bias[0].unsqueeze(0) if dt_projs_bias is not None else None
        K, D, R = dt_projs_weight.shape # K=1
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # xs = CrossScan.apply(x)
        xs = x.view(B, -1, L).unsqueeze(dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, L)

        # y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)

        return y


    def cross_selective_scan_multimodal_k1(
        x_rgb: torch.Tensor=None,
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 1, C, 2 * H * W
        B, C, H, W = x_rgb.shape
        x_fuse = x_rgb.new_empty((B, 1, C, 2 * H * W))
        x_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2*H*W)

        # y = CrossMerge_multimodal.apply(ys)
        y = ys[:, 0, :, 0:L//2] + ys[:, 0, :, L//2:L]

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x_rgb.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x_rgb.dtype)

        return y

    def cross_selective_scan_multimodal_k2(
        x_rgb: torch.Tensor=None,
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm1: torch.nn.Module=None,
        out_norm2: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 2, C, 2 * H * W

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2*H*W)

        y_rgb, y_e = CrossMerge_multimodal.apply(ys)

        y_rgb = y_rgb.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_e = y_e.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_rgb = out_norm1(y_rgb).to(x_rgb.dtype)
        y_e = out_norm2(y_e).to(x_e.dtype)

        return y_rgb, y_e


# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop


    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


def print_jit_input_names(inputs):
    # tensor.11, dt.1, A.1, B.1, C.1, D.1, z.1, None
    try:
        print("input params: ", end=" ", flush=True)
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)

    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[1].debugName().startswith("dts") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    assert inputs[4].debugName().startswith("Cs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = len(inputs) > 5 and inputs[5].debugName().startswith("z")
    else:
        with_z = len(inputs) > 6 and inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    # flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops

# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


DEV = False
class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv
        d_conv=3, # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        if DEV:
            d_conv = -1

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 4 if not (self.forward_core == self.forward_corev1_share_ssm) else 1
        # VMamaba set K=4, while original SSM set K=1
        if self.forward_core == self.forward_corev0:
            self.K = 1
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.K2 = self.K if not (self.forward_core == self.forward_corev1_share_a) else 1
        # VMamaba set K=4, while original SSM set K=1
        if self.forward_core == self.forward_corev0:
            self.K2 = 1
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 1 #4

        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        xs = x.view(B, -1, L).unsqueeze(dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, #z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            # return_last_state=False,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        # inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = torch.transpose(out_y[:, 0], dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev0_seq(self, x: torch.Tensor):
        selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float() # (b, k, d, l)
        dts = dts.contiguous().float() # (b, k, d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1) # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1) # (k, d)

        # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev1(self, x: torch.Tensor, float32=True):
        # float32 should be true in training!!!! otherwise, the output of selective_scan would be inf...
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        xs = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        As = -torch.exp(self.A_logs.to(torch.float))  # (k * d, d_state)
        Ds = self.Ds.to(torch.float) # (k * d)
        dt_projs_bias = self.dt_projs_bias.to(torch.float).view(-1) # (k * d)

        if float32:
            ys: torch.Tensor = selective_scan(
                xs.to(torch.float),
                dts.to(torch.float),
                As,
                Bs.to(torch.float),
                Cs.to(torch.float),
                Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        else:
            out_y: torch.Tensor = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            # assert out_y.dtype == torch.float16

            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()

        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)

            # if torch.isinf(y).any() or torch.isnan(y).any():
            #     for item in [y, xs, dts, As, Bs, Cs, Ds]:
            #         print(torch.isinf(item).any(), torch.isnan(item).any(), item.max(), item.min())
            #     import time; time.sleep(10000)

        return y

    def forward_corev1_share_ssm(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
            return xs

        x_dbl = torch.einsum("b d l, c d -> b c l", x.view(B, -1, L), self.x_proj_weight[0])
        # x_dbl = x_dbl + self.x_proj_bias.view(1, -1, 1)
        dt, BC = torch.split(x_dbl, [self.dt_rank, 2 * self.d_state], dim=1)
        dt = torch.einsum("b r l, d r -> b d l", dt, self.dt_projs_weight[0])
        x_dt_BC = torch.cat([x, dt.view(B, -1, H, W), BC.view(B, -1, H, W)], dim=1) # (b, -1, h, w)

        x_dt_BCs = cross_scan_2d(x_dt_BC) # (b, k, d, l)
        xs, dts, Bs, Cs = torch.split(x_dt_BCs, [self.d_inner, self.d_inner, self.d_state, self.d_state], dim=2)

        xs = xs.contiguous().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        As = -torch.exp(self.A_logs.float()).repeat(4, 1) # (k * d, d_state)
        Ds = self.Ds.repeat(4) # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1).repeat(4) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 4, -1, L)
        # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()

        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1_share_a(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x, dim=1):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=dim)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=dim) # (b, k, d, l)
            return xs

        K = 4
        xs = cross_scan_2d(x, dim=1) # (b, d, k, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        dts = dts + self.dt_projs_bias.to(xs.dtype).view(1, K, -1, 1)

        xs = xs.transpose(dim0=1, dim1=2).contiguous().view(B, -1, K * L)
        dts = dts.transpose(dim0=1, dim1=2).contiguous().view(B, -1, K * L)
        As = -torch.exp(self.A_logs.float()) # (D, N)
        Ds = self.Ds.view(-1) # (D)
        Bs = Bs.transpose(dim0=1, dim1=2).contiguous().view(B, 1, -1, K * L)
        Cs = Cs.transpose(dim0=1, dim1=2).contiguous().view(B, 1, -1, K * L)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=None,
            delta_softplus=True,
        ).view(B, -1, 4, L)
        # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, :, 2:4], dims=[-1]).view(B, -1, 2, L)
        wh_y = torch.transpose(out_y[:, :, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, :, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, :, 0].float() + inv_y[:, :, 0].float() + wh_y.float() + invwh_y.float()

        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev2(self, x: torch.Tensor, nrows=-1):
        return cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version,
            nrows=nrows,
        )

    def forward_core_1d(self, x: torch.Tensor):
        return selective_scan_1d(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version,
        )

    # forward_core = forward_core_share_ssm
    # forward_core = forward_core_share_a
    # forward_core = forward_corev1
    forward_core = forward_corev2 # vmamba
    # forward_core = forward_corev0 # ori mamba
    # forward_core = forward_core_1d # ori mamba

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x)) # (b, d, h, w)
            y = self.forward_core(x)
            if self.softmax_version:
                y = y * z
            else:
                y = y * F.silu(z)
        else:
            if self.softmax_version:
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
                x = F.silu(x)
            else:
                xz = F.silu(xz)
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            y = self.forward_core(x)
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class ConMB_SS2D(nn.Module):
    '''
    Multimodal Mamba Selective Scan 2D
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=4,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv
        d_conv=3, # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        if DEV:
            d_conv = -1

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_modalx = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 2
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm1 = nn.LayerNorm(self.d_inner)
            self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner*2, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2_multimodal(self, x_rgb: torch.Tensor, x_e: torch.Tensor, nrows=-1):
        return cross_selective_scan_multimodal_k2(
            x_rgb, x_e, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm1", None), getattr(self, "out_norm2", None), self.softmax_version,
            nrows=nrows,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans)) # (b, d, h, w)
            x_e_conv = self.act(self.conv2d_modalx(x_e_trans)) # (b, d, h, w)
            y_rgb, y_e = self.forward_corev2_multimodal(x_rgb_conv, x_e_conv) # b, d, h, w -> b, h, w, d
            # SE to get attention, scale
            b, d, h, w = x_rgb_trans.shape
            x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
            x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)
            x_rgb_exitation = self.fc1(x_rgb_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous() # b, 1, 1, d
            x_e_exitation = self.fc2(x_e_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()
            y_rgb = y_rgb * x_e_exitation
            y_e = y_e * x_rgb_exitation
            y = torch.concat([y_rgb, y_e], dim=-1)
        out = self.dropout(self.out_proj(y))
        return out

# =====================================================
class SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.D = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x.shape
        x = x.permute(0, 2, 1)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        A = -torch.exp(self.A_log.float())  # (k * d, d_state)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        y = selective_scan(
            x, dt,
            A, B, C, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y = rearrange(y, "b d l -> b l d")
        y = self.out_norm(y)
        return y


# =====================================================
class Cross_Mamba_Attention_SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.D_1 = self.D_init(self.d_inner)  # (D)
        self.D_2 = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x_rgb.shape
        x_rgb = x_rgb.permute(0, 2, 1)
        x_e = x_e.permute(0, 2, 1)
        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))  # (bl d)
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))  # (bl d)
        dt_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()
        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", l=L)
        dt_e = rearrange(dt_e, "d (b l) -> b d l", l=L)
        A_rgb = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        A_e = -torch.exp(self.A_log_2.float())  # (k * d, d_state)
        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", l=L).contiguous()

        y_rgb = selective_scan(
            x_rgb, dt_rgb,
            A_rgb, B_rgb, C_e, self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(),
            delta_softplus=True,
        )
        y_e = selective_scan(
            x_e, dt_e,
            A_e, B_e, C_rgb, self.D_2.float(),
            delta_bias=self.dt_proj_2.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y_rgb = rearrange(y_rgb, "b d l -> b l d")
        y_rgb = self.out_norm_1(y_rgb)
        y_e = rearrange(y_e, "b d l -> b l d")
        y_e = self.out_norm_2(y_e)
        return y_rgb, y_e


# =====================================================
class CrossMambaFusion_SS2D_SSM(nn.Module):
    '''
    Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv
        d_conv=3, # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.CMA_ssm = Cross_Mamba_Attention_SSM(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            **kwargs,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        B, H, W, D = x_rgb.shape
        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans)) # (b, d, h, w)
            x_e_conv = self.act(self.conv2d(x_e_trans)) # (b, d, h, w)
            x_rgb_conv = rearrange(x_rgb_conv, "b d h w -> b (h w) d")
            x_e_conv = rearrange(x_e_conv, "b d h w -> b (h w) d")
            y_rgb, y_e = self.CMA_ssm(x_rgb_conv, x_e_conv)
            # to b, d, h, w
            y_rgb = y_rgb.view(B, H, W, -1)
            y_e = y_e.view(B, H, W, -1)

        out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
        out_e = self.dropout_e(self.out_proj_e(y_e))
        return out_rgb, out_e


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        return x * self.sigmoid(attn)


class ChannelAttentionBlock(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(ChannelAttentionBlock, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class CVSSDecoderBlock(nn.Module):
    '''
    Channel-Aware VSS
    '''
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.scale1 = nn.Parameter(torch.ones(hidden_dim))
        self.op = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        self.conv_blk = ChannelAttentionBlock(hidden_dim)
        self.norm2 = norm_layer(hidden_dim)
        self.scale2 = nn.Parameter(torch.ones(hidden_dim))

    def _forward(self, input: torch.Tensor):
        x = input*self.scale1 + self.drop_path(self.op(self.norm1(input)))

        y = self.conv_blk(self.norm2(x).permute(0, 3, 1, 2).contiguous()) + (x*self.scale2).permute(0, 3, 1, 2).contiguous()
        y = y.permute(0, 2, 3, 1).contiguous()
        return y

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class CrossMambaFusionBlock(nn.Module):
    '''
    Cross Mamba Fusion (CroMB) fusion, with 2d SSM
    '''
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = CrossMambaFusion_SS2D_SSM(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb_cross, x_e_cross = self.op(x_rgb, x_e)
        x_rgb = x_rgb + self.drop_path1(x_rgb_cross)
        x_e = x_e + self.drop_path2(x_e_cross)
        return x_rgb, x_e

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)


class ConcatMambaFusionBlock(nn.Module):
    '''
    Concat Mamba (ConMB) fusion, with 2d SSM
    '''
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = ConMB_SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x = x_rgb + x_e + self.drop_path(self.op(x_rgb, x_e))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)


class VSSM(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            # =========================
            d_state=16,
            dt_rank="auto",
            ssm_ratio=2.0,
            attn_drop_rate=0.,
            shared_ssm=False,
            softmax_version=False,
            # =========================
            drop_rate=0.,
            drop_path_rate=0.1,
            mlp_ratio=4.0,
            patch_norm=True,
            norm_layer=nn.LayerNorm,
            downsample_version: str = "v2",
            use_checkpoint=False,
            **kwargs,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, self.embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(self.embed_dim) if patch_norm else nn.Identity()),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            if downsample_version == "v2":
                downsample = self._make_downsample(
                    self.dims[i_layer],
                    self.dims[i_layer + 1],
                    norm_layer=norm_layer,
                ) if (i_layer < self.num_layers - 1) else nn.Identity()
            else:
                downsample = PatchMerging2D(
                    self.dims[i_layer],
                    self.dims[i_layer + 1],
                    norm_layer=norm_layer,
                ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                depth = depths[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                attn_drop_rate=attn_drop_rate,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96,
        depth=2,
        drop_path=[0.1, 0.1],
        use_checkpoint=False,
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        # ===========================
        d_state=16,
        dt_rank="auto",
        ssm_ratio=2.0,
        attn_drop_rate=0.0,
        shared_ssm=False,
        softmax_version=False,
        # ===========================
        mlp_ratio=4.0,
        drop_rate=0.0,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
                **kwargs,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.CrossScan": None,
            "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")
        # print('after:', state_dict.keys())

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 depths=[2, 2, 9, 2], dims=[96, 192, 384, 768],
                 d_state=16, ssm_ratio=2.0, attn_drop_rate=0.,
                 drop_rate=0., drop_path_rate=0.1, mlp_ratio=4.0,
                 patch_norm=True, norm_layer=nn.LayerNorm,
                 downsample_version: str = "v2",
                 use_checkpoint=False,
                 out_indices=(0, 1, 2, 3), pretrained=None,
                 **kwargs,
        ):
        super().__init__(patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         depths=depths, dims=dims,
                         d_state=d_state, ssm_ratio=ssm_ratio, attn_drop_rate=attn_drop_rate,
                         drop_rate=drop_rate, drop_path_rate=drop_path_rate, mlp_ratio=mlp_ratio,
                         patch_norm=patch_norm, norm_layer=norm_layer,
                         downsample_version=downsample_version,
                         use_checkpoint=use_checkpoint,
                         **kwargs)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            # print(f"ckpt keys: {_ckpt['model'].keys()}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print('incompatible:', incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x

        return outs

##----------------------------------------------------My Experriment----------------------------------------------##
##----------------------------------------------------My Experriment----------------------------------------------##
##-------------------------------------------------------My CrosMB------------------------------------------------##
##-------------------------------------------Thermal与RGB互相指导离散化步长  德尔塔---------------------------------##

# 1. 核心数学层：加入昼夜门控步长调制
class CrossModalStateCore(nn.Module):
    def __init__(self, d_model=96, d_state=4, ssm_ratio=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, **kwargs):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)

        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)
        self.D_1 = self.D_init(self.d_inner)
        self.D_2 = self.D_init(self.d_inner)

        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)

        # ====== 咱们的独家创新：跨模态步长门控层 ======
        self.dt_gate_rgb = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.dt_gate_e = nn.Linear(self.d_inner, self.d_inner, bias=False)
        # ============================================

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x_rgb.shape
        x_rgb = x_rgb.permute(0, 2, 1)
        x_e = x_e.permute(0, 2, 1)

        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))

        dt_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()
        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", l=L)
        dt_e = rearrange(dt_e, "d (b l) -> b d l", l=L)

        # ====== 咱们的独家创新：用另一模态去干预步长 ======
        gate_rgb = torch.sigmoid(self.dt_gate_rgb(rearrange(x_e, "b d l -> b l d")))
        gate_rgb = rearrange(gate_rgb, "b l d -> b d l")
        gate_e = torch.sigmoid(self.dt_gate_e(rearrange(x_rgb, "b d l -> b l d")))
        gate_e = rearrange(gate_e, "b l d -> b d l")

        dt_rgb = dt_rgb * gate_rgb
        dt_e = dt_e * gate_e
        # =================================================

        A_rgb = -torch.exp(self.A_log_1.float())
        A_e = -torch.exp(self.A_log_2.float())
        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", l=L).contiguous()

        y_rgb = selective_scan(
            x_rgb, dt_rgb, A_rgb, B_rgb, C_e, self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(), delta_softplus=True,
        )
        y_e = selective_scan(
            x_e, dt_e, A_e, B_e, C_rgb, self.D_2.float(),
            delta_bias=self.dt_proj_2.bias.float(), delta_softplus=True,
        )

        y_rgb = rearrange(y_rgb, "b d l -> b l d")
        y_rgb = self.out_norm_1(y_rgb)
        y_e = rearrange(y_e, "b d l -> b l d")
        y_e = self.out_norm_2(y_e)
        return y_rgb, y_e     ####1

# 2. 中间包装层：调用我们的 Ours 核心

class CrossModalStateOperator(nn.Module):
    def __init__(self, d_model=96, d_state=16, ssm_ratio=2, dt_rank="auto", d_conv=3, conv_bias=True, dropout=0., bias=False, dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, softmax_version=False, **kwargs):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner, bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs)
            self.act = nn.SiLU()

        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # ====== 这里换成了调用我们自己的类 ======
        self.CMA_ssm = CrossModalStateCore(
            d_model=self.d_model, d_state=self.d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank, dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor, **kwargs,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        B, H, W, D = x_rgb.shape
        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans))
            x_e_conv = self.act(self.conv2d(x_e_trans))
            x_rgb_conv = rearrange(x_rgb_conv, "b d h w -> b (h w) d")
            x_e_conv = rearrange(x_e_conv, "b d h w -> b (h w) d")
            #y_rgb, y_e, dt_rgb, dt_e = self.CMA_ssm(x_rgb_conv, x_e_conv)
            y_rgb, y_e = self.CMA_ssm(x_rgb_conv, x_e_conv)    ####2
            y_rgb = y_rgb.view(B, H, W, -1)
            y_e = y_e.view(B, H, W, -1)

        out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
        out_e = self.dropout_e(self.out_proj_e(y_e))
        return out_rgb, out_e ####3

# 3. 最外层模块：暴露给外部骨架的 API
class CrossModalStateBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), attn_drop_rate: float = 0, d_state: int = 4, dt_rank: Any = "auto", ssm_ratio=2.0, shared_ssm=False, softmax_version=False, use_checkpoint: bool = False, mlp_ratio=0.0, act_layer=nn.GELU, drop: float = 0.0, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # ====== 这里换成了调用我们自己的中间层 ======
        self.op = CrossModalStateOperator(
            d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank, shared_ssm=shared_ssm, softmax_version=softmax_version, **kwargs
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb_cross, x_e_cross = self.op(x_rgb, x_e) ####4
        #x_rgb_cross, x_e_cross, dt_rgb, dt_e = self.op(x_rgb, x_e)
        x_rgb = x_rgb + self.drop_path1(x_rgb_cross)
        x_e = x_e + self.drop_path2(x_e_cross)
        return x_rgb, x_e ####5

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        else:
            return self._forward(x_rgb, x_e)


# =====================================================================
# DWR-Mamba low-level modules
#
# v3 changes:
#   1. Use the memory-safe cross-modal state interaction implementation.
#   2. Add CCS calibration to the two hidden cross-scale gates in ConMB.
#
# 中文：
#   1）CroMB 侧不重新发明，沿用 v2 的 memory-safe router + gate^s，
#      但用 V3 wrapper 让主网络命名干净。
#   2）ConMB 侧严格保留原版 Concat Selective Scan 主体，
#      只在两个 hidden cross-scale gate 后加相关性感知校准。
# =====================================================================


class RCSFCalibration(nn.Module):
    """
    Correlation-aware Cross-Scale Calibration.

    Input:
        x_rgb: [B, D, H, W]
        x_e:   [B, D, H, W]

    Output:
        calib_t2r: [B, D]  calibration for Thermal -> RGB scale
        calib_r2t: [B, D]  calibration for RGB -> Thermal scale

    Notes:
        - channel-level only, no spatial gate;
        - no additional selective scan;
        - statistics are detached by default for memory safety;
        - zero-init final layer, so initial output is exactly 1.
    """

    def __init__(
        self,
        d_inner: int,
        reduction: int = 16,
        min_hidden: int = 16,
        corr_scale: float = 0.1,
        detach_stats: bool = True,
        use_feat_stats: bool = True,
        use_diff_stat: bool = True,
        use_coact_stat: bool = True,
    ):
        super().__init__()

        self.d_inner = d_inner
        self.detach_stats = detach_stats
        self.corr_scale = corr_scale
        self.use_feat_stats = use_feat_stats
        self.use_diff_stat = use_diff_stat
        self.use_coact_stat = use_coact_stat

        hidden = max(d_inner // reduction, min_hidden)

        self.norm = nn.LayerNorm(4 * d_inner)
        self.fc1 = nn.Linear(4 * d_inner, hidden, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(hidden, 2 * d_inner, bias=True)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)

        # zero-init: calib_t2r = calib_r2t = 1 at initialization
        nn.init.constant_(self.fc2.weight, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)

    @staticmethod
    def _gap(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))

    def _compute_stats(self, x_rgb: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        """
        Compute [GAP(R), GAP(T), GAP(|R-T|), GAP(R*T)].
        """
        x_rgb_f = x_rgb.float()
        x_e_f = x_e.float()

        if self.use_feat_stats:
            rgb_stat = self._gap(x_rgb_f)
            e_stat = self._gap(x_e_f)
        else:
            rgb_stat = torch.zeros(
                x_rgb_f.shape[0],
                x_rgb_f.shape[1],
                device=x_rgb_f.device,
                dtype=x_rgb_f.dtype,
            )
            e_stat = torch.zeros_like(rgb_stat)

        if self.use_diff_stat:
            diff_stat = self._gap((x_rgb_f - x_e_f).abs())
        else:
            diff_stat = torch.zeros_like(rgb_stat)

        if self.use_coact_stat:
            prod_stat = self._gap(x_rgb_f * x_e_f)
        else:
            prod_stat = torch.zeros_like(rgb_stat)

        z = torch.cat([rgb_stat, e_stat, diff_stat, prod_stat], dim=-1)

        del x_rgb_f, x_e_f, rgb_stat, e_stat, diff_stat, prod_stat
        return z

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        dtype = x_rgb.dtype
        device = x_rgb.device

        if self.detach_stats:
            with torch.no_grad():
                z = self._compute_stats(x_rgb, x_e)
        else:
            z = self._compute_stats(x_rgb, x_e)

        z = z.to(device=device, dtype=torch.float32)

        delta = self.fc2(self.act(self.fc1(self.norm(z))))
        delta_t2r, delta_r2t = delta.chunk(2, dim=-1)

        calib_t2r = 1.0 + self.corr_scale * torch.tanh(delta_t2r)
        calib_r2t = 1.0 + self.corr_scale * torch.tanh(delta_r2t)

        return calib_t2r.to(dtype=dtype), calib_r2t.to(dtype=dtype)


class RCSFSelectiveScan(ConMB_SS2D):
    """
    CCS version of original ConMB_SS2D.

    原版 ConMB:
        base_rgb = fc1(GAP(rgb))
        base_e   = fc2(GAP(e))

        y_rgb = y_rgb * base_e
        y_e   = y_e   * base_rgb

    CCS-ConMB:
        scale_t2r = base_e   * calib_t2r
        scale_r2t = base_rgb * calib_r2t

        y_rgb = y_rgb * scale_t2r
        y_e   = y_e   * scale_r2t

    其余 Linear / DWConv / Concat SS / concat / out_proj 全部保持原版流程。
    """

    def __init__(
        self,
        d_model=96,
        d_state=4,
        ssm_ratio=2,
        dt_rank="auto",
        d_conv=3,
        conv_bias=True,
        dropout=0.,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        softmax_version=False,
        ccs_reduction: int = 16,
        ccs_min_hidden: int = 16,
        ccs_scale: float = 0.1,
        detach_ccs_stats: bool = True,
        use_rcsf: bool = True,
        use_rcsf_feat: bool = True,
        use_rcsf_diff: bool = True,
        use_rcsf_coact: bool = True,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            bias=bias,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            softmax_version=softmax_version,
            **kwargs,
        )

        self.ccs_calibration = RCSFCalibration(
            d_inner=self.d_inner,
            reduction=ccs_reduction,
            min_hidden=ccs_min_hidden,
            corr_scale=ccs_scale,
            detach_stats=detach_ccs_stats,
            use_feat_stats=use_rcsf_feat,
            use_diff_stat=use_rcsf_diff,
            use_coact_stat=use_rcsf_coact,
        )
        self.use_rcsf = use_rcsf

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        # -------------------------------------------------------------
        # Original ConMB path: projection
        # -------------------------------------------------------------
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)

        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()

            x_rgb_conv = self.act(self.conv2d(x_rgb_trans))
            x_e_conv = self.act(self.conv2d_modalx(x_e_trans))

            # Original Concat Selective Scan
            y_rgb, y_e = self.forward_corev2_multimodal(
                x_rgb_conv,
                x_e_conv,
            )

            # ---------------------------------------------------------
            # Original two hidden cross-scale gates
            # ---------------------------------------------------------
            b, d, h, w = x_rgb_trans.shape

            x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
            x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)

            base_rgb = self.fc1(x_rgb_squeeze)  # RGB descriptor, used for RGB -> T
            base_e = self.fc2(x_e_squeeze)      # Thermal descriptor, used for T -> RGB

            # ---------------------------------------------------------
            # New CCS calibration
            # ---------------------------------------------------------
            calib_t2r, calib_r2t = self.ccs_calibration(
                x_rgb_trans,
                x_e_trans,
            )
            if self.use_rcsf:
                scale_t2r = base_e * calib_t2r       # Thermal -> RGB
                scale_r2t = base_rgb * calib_r2t     # RGB -> Thermal
            else:
                scale_t2r = base_e + (calib_t2r - 1.0) * 0.0
                scale_r2t = base_rgb + (calib_r2t - 1.0) * 0.0

            scale_t2r = scale_t2r.view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()
            scale_r2t = scale_r2t.view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()

            y_rgb = y_rgb * scale_t2r
            y_e = y_e * scale_r2t

            y = torch.concat([y_rgb, y_e], dim=-1)

        out = self.dropout(self.out_proj(y))
        return out


class RCSFBlock(ConcatMambaFusionBlock):
    """
    Wrapper of CCS-ConMB.

    It keeps the original ConcatMambaFusionBlock residual structure
    and only replaces:
        ConMB_SS2D -> RCSFSelectiveScan
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        ccs_reduction: int = 16,
        ccs_min_hidden: int = 16,
        ccs_scale: float = 0.1,
        detach_ccs_stats: bool = True,
        use_rcsf: bool = True,
        use_rcsf_feat: bool = True,
        use_rcsf_diff: bool = True,
        use_rcsf_coact: bool = True,
        **kwargs,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            dt_rank=dt_rank,
            ssm_ratio=ssm_ratio,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            use_checkpoint=use_checkpoint,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            drop=drop,
            **kwargs,
        )

        self.use_rcsf = bool(use_rcsf)

        self.op = RCSFSelectiveScan(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            ccs_reduction=ccs_reduction,
            ccs_min_hidden=ccs_min_hidden,
            ccs_scale=ccs_scale,
            detach_ccs_stats=detach_ccs_stats,
            use_rcsf=use_rcsf,
            use_rcsf_feat=use_rcsf_feat,
            use_rcsf_diff=use_rcsf_diff,
            use_rcsf_coact=use_rcsf_coact,
            **kwargs,
        )

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        if not self.use_rcsf:
            return x_rgb + x_e
        return super()._forward(x_rgb, x_e)






# =====================================================================
# WRCSI and BWM implementation
# Reliability-aware bidirectional state interaction, memory-safe version
#
# 目标：
#   1. 保留 V2 的主体思想；
#   2. 保留四个 stage 全部使用 WRCSI；
#   3. 保留 feature/gate/dt/top-k 统计；
#   4. 保留 m = gate ^ s 的幂次调制；
#   5. 训练时减少 router 统计分支的反向图显存。
#
# 关键变化：
#   Router 的统计量从 x/gate/dt 中提取，但默认 detach。
#   这样：
#       - router 的 fc/norm 参数仍然可训练；
#       - s_t2r/s_r2t 仍然参与 gate^s；
#       - gate^s 主路径仍然对 gate 和 dt 反传；
#       - 但不会再为了统计 mean/std/corr/topk 保存一大堆反向图。
#
# 中文：
#   这不是砍掉 V2，而是把 V2 写成 small 可训练的显存安全版。
# =====================================================================


class BWMRouter(nn.Module):
    """
    Step-dynamics and local-saliency aware direction-strength router.

    Inputs:
        x_rgb:    [B, D, L]
        x_e:      [B, D, L]
        gate_rgb: [B, D, L]  gate generated from Thermal/X to modulate RGB
        gate_e:   [B, D, L]  gate generated from RGB to modulate Thermal/X
        dt_rgb:   [B, D, L]  raw dt before final gate modulation
        dt_e:     [B, D, L]

    Outputs:
        s_t2r: [B, D, 1]
        s_r2t: [B, D, 1]
    """

    def __init__(
        self,
        d_inner: int,
        hidden: int = 32,
        s_min: float = 0.25,
        s_max: float = 1.75,
        topk_ratio: float = 0.05,
        eps: float = 1e-6,
        init_std: float = 0.02,
        detach_stats: bool = True,
        bwm_mode: str = "full",
        bwm_stage=None,
    ):
        super().__init__()

        self.bwm_mode = validate_bwm_mode(bwm_mode)
        self.bwm_stage = bwm_stage

        assert s_min < 1.0 < s_max, \
            f"s_min < 1 < s_max is required, got s_min={s_min}, s_max={s_max}"

        self.d_inner = d_inner
        self.s_min = s_min
        self.s_max = s_max
        self.topk_ratio = topk_ratio
        self.eps = eps
        self.detach_stats = detach_stats

        stat_dim = 19

        self.norm = nn.LayerNorm(stat_dim)
        self.fc1 = nn.Linear(stat_dim, hidden, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(hidden, 2, bias=True)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.normal_(self.fc2.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.fc2.bias, mean=0.0, std=init_std)

    def _topk_mean(self, x: torch.Tensor):
        """
        x: [B, D, L]
        return: [B, D]
        """
        B, D, L = x.shape
        k = max(1, int(L * self.topk_ratio))
        return torch.topk(x, k=k, dim=-1).values.mean(dim=-1)

    def _compute_stats_no_grad(
        self,
        x_rgb: torch.Tensor,
        x_e: torch.Tensor,
        gate_rgb: torch.Tensor,
        gate_e: torch.Tensor,
        dt_rgb: torch.Tensor,
        dt_e: torch.Tensor,
    ):
        """
        统计分支显存安全版。

        注意：
            这里算出的 z 只作为 router MLP 的输入；
            z 不需要对 x/gate/dt 反传梯度。
            这样能显著减少 small 上的激活显存。
        """
        x_rgb_f = x_rgb.detach().float()
        x_e_f = x_e.detach().float()
        gate_rgb_f = gate_rgb.detach().float()
        gate_e_f = gate_e.detach().float()
        dt_rgb_f = dt_rgb.detach().float()
        dt_e_f = dt_e.detach().float()

        # -------------------------------------------------------------
        # 1. Feature relation statistics
        # -------------------------------------------------------------
        mean_rgb = x_rgb_f.mean(dim=-1)
        mean_e = x_e_f.mean(dim=-1)

        std_rgb = x_rgb_f.var(dim=-1, unbiased=False).add(self.eps).sqrt()
        std_e = x_e_f.var(dim=-1, unbiased=False).add(self.eps).sqrt()

        mean_abs_feat_diff = (x_rgb_f - x_e_f).abs().mean(dim=-1)
        mean_feat_product = (x_rgb_f * x_e_f).mean(dim=-1)

        x_rgb_center = x_rgb_f - mean_rgb.unsqueeze(-1)
        x_e_center = x_e_f - mean_e.unsqueeze(-1)

        corr_num = (x_rgb_center * x_e_center).mean(dim=-1)
        corr_den = (
            x_rgb_center.pow(2).mean(dim=-1).add(self.eps).sqrt()
            * x_e_center.pow(2).mean(dim=-1).add(self.eps).sqrt()
        )
        feat_corr = corr_num / corr_den

        # 用完大中间量及时释放引用
        del x_rgb_center, x_e_center, corr_num, corr_den

        # -------------------------------------------------------------
        # 2. Gate statistics
        # -------------------------------------------------------------
        mean_gate_rgb = gate_rgb_f.mean(dim=-1)
        mean_gate_e = gate_e_f.mean(dim=-1)

        std_gate_rgb = gate_rgb_f.var(dim=-1, unbiased=False).add(self.eps).sqrt()
        std_gate_e = gate_e_f.var(dim=-1, unbiased=False).add(self.eps).sqrt()

        mean_abs_gate_diff = (gate_rgb_f - gate_e_f).abs().mean(dim=-1)

        # -------------------------------------------------------------
        # 3. dt dynamics statistics
        # -------------------------------------------------------------
        mean_abs_dt_rgb = torch.log1p(dt_rgb_f.abs()).mean(dim=-1)
        mean_abs_dt_e = torch.log1p(dt_e_f.abs()).mean(dim=-1)
        mean_abs_dt_diff = torch.log1p((dt_rgb_f - dt_e_f).abs()).mean(dim=-1)

        # -------------------------------------------------------------
        # 4. Local saliency statistics
        # -------------------------------------------------------------
        topk_abs_rgb = self._topk_mean(x_rgb_f.abs())
        topk_abs_e = self._topk_mean(x_e_f.abs())

        topk_gate_rgb = self._topk_mean(gate_rgb_f)
        topk_gate_e = self._topk_mean(gate_e_f)

        z = torch.stack(
            [
                mean_rgb,
                mean_e,
                std_rgb,
                std_e,
                mean_abs_feat_diff,
                mean_feat_product,
                feat_corr,
                mean_gate_rgb,
                mean_gate_e,
                std_gate_rgb,
                std_gate_e,
                mean_abs_gate_diff,
                mean_abs_dt_rgb,
                mean_abs_dt_e,
                mean_abs_dt_diff,
                topk_abs_rgb,
                topk_abs_e,
                topk_gate_rgb,
                topk_gate_e,
            ],
            dim=-1,
        )  # [B, D, 19]

        return z

    def _compute_stats_with_grad(
        self,
        x_rgb: torch.Tensor,
        x_e: torch.Tensor,
        gate_rgb: torch.Tensor,
        gate_e: torch.Tensor,
        dt_rgb: torch.Tensor,
        dt_e: torch.Tensor,
    ):
        """
        原始可反传统计版本。
        默认不用它，只保留备用。
        """
        x_rgb_f = x_rgb.float()
        x_e_f = x_e.float()
        gate_rgb_f = gate_rgb.float()
        gate_e_f = gate_e.float()
        dt_rgb_f = dt_rgb.float()
        dt_e_f = dt_e.float()

        mean_rgb = x_rgb_f.mean(dim=-1)
        mean_e = x_e_f.mean(dim=-1)

        std_rgb = x_rgb_f.var(dim=-1, unbiased=False).add(self.eps).sqrt()
        std_e = x_e_f.var(dim=-1, unbiased=False).add(self.eps).sqrt()

        mean_abs_feat_diff = (x_rgb_f - x_e_f).abs().mean(dim=-1)
        mean_feat_product = (x_rgb_f * x_e_f).mean(dim=-1)

        x_rgb_center = x_rgb_f - mean_rgb.unsqueeze(-1)
        x_e_center = x_e_f - mean_e.unsqueeze(-1)

        corr_num = (x_rgb_center * x_e_center).mean(dim=-1)
        corr_den = (
            x_rgb_center.pow(2).mean(dim=-1).add(self.eps).sqrt()
            * x_e_center.pow(2).mean(dim=-1).add(self.eps).sqrt()
        )
        feat_corr = corr_num / corr_den

        mean_gate_rgb = gate_rgb_f.mean(dim=-1)
        mean_gate_e = gate_e_f.mean(dim=-1)

        std_gate_rgb = gate_rgb_f.var(dim=-1, unbiased=False).add(self.eps).sqrt()
        std_gate_e = gate_e_f.var(dim=-1, unbiased=False).add(self.eps).sqrt()

        mean_abs_gate_diff = (gate_rgb_f - gate_e_f).abs().mean(dim=-1)

        mean_abs_dt_rgb = torch.log1p(dt_rgb_f.abs()).mean(dim=-1)
        mean_abs_dt_e = torch.log1p(dt_e_f.abs()).mean(dim=-1)
        mean_abs_dt_diff = torch.log1p((dt_rgb_f - dt_e_f).abs()).mean(dim=-1)

        topk_abs_rgb = self._topk_mean(x_rgb_f.abs())
        topk_abs_e = self._topk_mean(x_e_f.abs())

        topk_gate_rgb = self._topk_mean(gate_rgb_f)
        topk_gate_e = self._topk_mean(gate_e_f)

        z = torch.stack(
            [
                mean_rgb,
                mean_e,
                std_rgb,
                std_e,
                mean_abs_feat_diff,
                mean_feat_product,
                feat_corr,
                mean_gate_rgb,
                mean_gate_e,
                std_gate_rgb,
                std_gate_e,
                mean_abs_gate_diff,
                mean_abs_dt_rgb,
                mean_abs_dt_e,
                mean_abs_dt_diff,
                topk_abs_rgb,
                topk_abs_e,
                topk_gate_rgb,
                topk_gate_e,
            ],
            dim=-1,
        )

        return z

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_e: torch.Tensor,
        gate_rgb: torch.Tensor,
        gate_e: torch.Tensor,
        dt_rgb: torch.Tensor,
        dt_e: torch.Tensor,
    ):
        if self.detach_stats:
            # 只关闭统计分支对 x/gate/dt 的反向图；
            # router 的 norm/fc 仍然正常训练。
            with torch.no_grad():
                z = self._compute_stats_no_grad(
                    x_rgb=x_rgb,
                    x_e=x_e,
                    gate_rgb=gate_rgb,
                    gate_e=gate_e,
                    dt_rgb=dt_rgb,
                    dt_e=dt_e,
                )
        else:
            z = self._compute_stats_with_grad(
                x_rgb=x_rgb,
                x_e=x_e,
                gate_rgb=gate_rgb,
                gate_e=gate_e,
                dt_rgb=dt_rgb,
                dt_e=dt_e,
            )

        # 注意：
        # z 不需要 grad 也没问题，fc/norm 参数仍然会得到梯度。
        z = self.norm(z)
        r = self.fc2(self.act(self.fc1(z)))  # [B, D, 2]

        s = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(r)

        # Preserve the original bounded exponents in full mode.
        if self.bwm_mode == "shared_directions":
            s = s.mean(dim=-1, keepdim=True).expand_as(s)
        elif self.bwm_mode == "shared_channels":
            s = s.mean(dim=1, keepdim=True).expand_as(s)

        s_t2r = s[..., 0].unsqueeze(-1).to(dtype=x_rgb.dtype, device=x_rgb.device)
        s_r2t = s[..., 1].unsqueeze(-1).to(dtype=x_rgb.dtype, device=x_rgb.device)

        return s_t2r, s_r2t


# Compatibility alias; does not register another module or parameter.
HBWRRouter = BWMRouter


class WRCSIStateInteractionCore(CrossModalStateCore):
    """
    WRCSI state interaction core, memory-safe version.

    Original cross-modal state path:
        dt_rgb = dt_rgb * gate_rgb
        dt_e   = dt_e   * gate_e

    WRCSI:
        s_t2r, s_r2t = Router(x_rgb, x_e, gate_rgb, gate_e, dt_rgb, dt_e)

        m_rgb = gate_rgb ^ s_t2r
        m_e   = gate_e   ^ s_r2t

        dt_rgb = dt_rgb * m_rgb
        dt_e   = dt_e   * m_e
    """

    def __init__(
        self,
        d_model=96,
        d_state=4,
        ssm_ratio=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        router_hidden: int = 32,
        router_init_std: float = 0.02,
        s_min: float = 0.25,
        s_max: float = 1.75,
        topk_ratio: float = 0.05,
        power_eps: float = 1e-4,
        detach_router_stats: bool = True,
        use_ri: bool = True,
        use_wi: bool = True,
        use_bwm: Optional[bool] = None,
        bwm_mode: str = "full",
        bwm_stage=None,
        **kwargs,
    ):
        use_bwm = resolve_bwm_enabled(use_bwm, kwargs.pop("use_hbwr", None))
        bwm_mode = validate_bwm_mode(bwm_mode)
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            **kwargs,
        )

        self.power_eps = power_eps
        self.use_ri = use_ri
        self.use_wi = use_wi
        self.use_bwm = use_bwm
        self.bwm_mode = bwm_mode
        self.bwm_stage = bwm_stage

        self.direction_router = BWMRouter(
            d_inner=self.d_inner,
            hidden=router_hidden,
            s_min=s_min,
            s_max=s_max,
            topk_ratio=topk_ratio,
            eps=1e-6,
            init_std=router_init_std,
            detach_stats=detach_router_stats,
            bwm_mode=bwm_mode,
            bwm_stage=bwm_stage,
        )

    @property
    def use_hbwr(self):
        """Legacy switch alias; the canonical switch is use_bwm."""
        return self.use_bwm

    @use_hbwr.setter
    def use_hbwr(self, enabled):
        self.use_bwm = bool(enabled)

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, L, _ = x_rgb.shape

        # [B, L, D] -> [B, D, L]
        x_rgb = x_rgb.permute(0, 2, 1).contiguous()
        x_e = x_e.permute(0, 2, 1).contiguous()

        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))

        dt_rgb, B_rgb, C_rgb = torch.split(
            x_dbl_rgb,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt_e, B_e, C_e = torch.split(
            x_dbl_e,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        # raw dt dynamic terms, before gate modulation and before softplus.
        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()

        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", b=B, l=L).contiguous()
        dt_e = rearrange(dt_e, "d (b l) -> b d l", b=B, l=L).contiguous()

        # -------------------------------------------------------------
        # 1. Original successful cross-modal step gate.
        #    不强制保留 gate_bld 的额外副本。
        # -------------------------------------------------------------
        if self.use_wi:
            gate_rgb = torch.sigmoid(
                self.dt_gate_rgb(x_e.transpose(1, 2))
            ).transpose(1, 2).contiguous()

            gate_e = torch.sigmoid(
                self.dt_gate_e(x_rgb.transpose(1, 2))
            ).transpose(1, 2).contiguous()
        else:
            gate_rgb_logits = self.dt_gate_rgb(x_e.transpose(1, 2)).transpose(1, 2)
            gate_e_logits = self.dt_gate_e(x_rgb.transpose(1, 2)).transpose(1, 2)
            gate_rgb = (torch.ones_like(dt_rgb) + gate_rgb_logits * 0.0).contiguous()
            gate_e = (torch.ones_like(dt_e) + gate_e_logits * 0.0).contiguous()

        # -------------------------------------------------------------
        # 2. Step-dynamics + local-saliency aware direction router.
        #
        # router 统计分支默认 detach，减少 small 训练激活显存。
        # -------------------------------------------------------------
        routed_s_t2r, routed_s_r2t = self.direction_router(
            x_rgb=x_rgb,
            x_e=x_e,
            gate_rgb=gate_rgb,
            gate_e=gate_e,
            dt_rgb=dt_rgb,
            dt_e=dt_e,
        )
        if self.use_bwm:
            s_t2r, s_r2t = routed_s_t2r, routed_s_r2t
        else:
            s_t2r = torch.ones_like(routed_s_t2r) + routed_s_t2r * 0.0
            s_r2t = torch.ones_like(routed_s_r2t) + routed_s_r2t * 0.0

        # -------------------------------------------------------------
        # 3. Power-gated step modulation.
        #
        # 原公式：
        #   m = g ^ s = exp(s * log(g))
        #
        # 这里用 torch.pow，避免显式保存 log/exp 多个大中间量。
        # -------------------------------------------------------------
        if self.use_bwm:
            m_rgb = torch.pow(gate_rgb.clamp_min(self.power_eps), s_t2r)
            m_e = torch.pow(gate_e.clamp_min(self.power_eps), s_r2t)
        else:
            m_rgb = gate_rgb + s_t2r * 0.0
            m_e = gate_e + s_r2t * 0.0

        emit_bwm_forward_evidence(
            self, s_t2r, s_r2t, m_rgb, m_e, gate_rgb, gate_e
        )

        dt_rgb = (dt_rgb * m_rgb).contiguous()
        dt_e = (dt_e * m_e).contiguous()

        # 用完就释放引用，降低后续峰值显存压力
        del m_rgb, m_e, s_t2r, s_r2t

        # -------------------------------------------------------------
        # Keep Sigma/CroMB cross-modal C readout unchanged.
        # RGB branch uses C_e.
        # Thermal branch uses C_rgb.
        # -------------------------------------------------------------
        A_rgb = -torch.exp(self.A_log_1.float())
        A_e = -torch.exp(self.A_log_2.float())

        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", b=B, l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", b=B, l=L).contiguous()
        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", b=B, l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", b=B, l=L).contiguous()

        y_rgb = selective_scan(
            x_rgb,
            dt_rgb,
            A_rgb,
            B_rgb,
            C_e if self.use_ri else C_rgb,
            self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(),
            delta_softplus=True,
        )

        y_e = selective_scan(
            x_e,
            dt_e,
            A_e,
            B_e,
            C_rgb if self.use_ri else C_e,
            self.D_2.float(),
            delta_bias=self.dt_proj_2.bias.float(),
            delta_softplus=True,
        )

        y_rgb = rearrange(y_rgb, "b d l -> b l d").contiguous()
        y_rgb = self.out_norm_1(y_rgb)

        y_e = rearrange(y_e, "b d l -> b l d").contiguous()
        y_e = self.out_norm_2(y_e)

        return y_rgb, y_e


class WRCSIStateInteractionSS2D(CrossModalStateOperator):
    """
    Wrapper of the WRCSI state interaction core.

    Input:
        x_rgb, x_e: [B, H, W, C]

    Output:
        out_rgb, out_e: [B, H, W, C]
    """

    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        d_conv=3,
        conv_bias=True,
        dropout=0.,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        softmax_version=False,
        router_hidden: int = 32,
        router_init_std: float = 0.02,
        s_min: float = 0.25,
        s_max: float = 1.75,
        topk_ratio: float = 0.05,
        power_eps: float = 1e-4,
        detach_router_stats: bool = True,
        use_ri: bool = True,
        use_wi: bool = True,
        use_bwm: Optional[bool] = None,
        bwm_mode: str = "full",
        bwm_stage=None,
        **kwargs,
    ):
        use_bwm = resolve_bwm_enabled(use_bwm, kwargs.pop("use_hbwr", None))
        bwm_mode = validate_bwm_mode(bwm_mode)
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            bias=bias,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            softmax_version=softmax_version,
            **kwargs,
        )

        self.CMA_ssm = WRCSIStateInteractionCore(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            router_hidden=router_hidden,
            router_init_std=router_init_std,
            s_min=s_min,
            s_max=s_max,
            topk_ratio=topk_ratio,
            power_eps=power_eps,
            detach_router_stats=detach_router_stats,
            use_ri=use_ri,
            use_wi=use_wi,
            use_bwm=use_bwm,
            bwm_mode=bwm_mode,
            bwm_stage=bwm_stage,
            **kwargs,
        )


class WRCSIBlock(CrossModalStateBlock):
    """
    WRCSI block.

    中文：
        全 stage 可用的新 CroMB。
        不替换 ConMB，只替换 CroMB 内部的步长调制机制。
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        router_hidden: int = 32,
        router_init_std: float = 0.02,
        s_min: float = 0.25,
        s_max: float = 1.75,
        topk_ratio: float = 0.05,
        power_eps: float = 1e-4,
        detach_router_stats: bool = True,
        use_ri: bool = True,
        use_wi: bool = True,
        use_bwm: Optional[bool] = None,
        bwm_mode: str = "full",
        bwm_stage=None,
        **kwargs,
    ):
        use_bwm = resolve_bwm_enabled(use_bwm, kwargs.pop("use_hbwr", None))
        bwm_mode = validate_bwm_mode(bwm_mode)
        super().__init__(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            dt_rank=dt_rank,
            ssm_ratio=ssm_ratio,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            use_checkpoint=use_checkpoint,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            drop=drop,
            **kwargs,
        )

        self.op = WRCSIStateInteractionSS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            router_hidden=router_hidden,
            router_init_std=router_init_std,
            s_min=s_min,
            s_max=s_max,
            topk_ratio=topk_ratio,
            power_eps=power_eps,
            detach_router_stats=detach_router_stats,
            use_ri=use_ri,
            use_wi=use_wi,
            use_bwm=use_bwm,
            bwm_mode=bwm_mode,
            bwm_stage=bwm_stage,
            **kwargs,
        )




# =====================================================================
# DCR-CroMB
# Delta-Coupled Readout Cross Mamba Block
#
# Core:
#   1. Keep your successful sigmoid delta modulation:
#          dt_rgb = dt_rgb * sigmoid(W_rgb(x_e))
#          dt_e   = dt_e   * sigmoid(W_e(x_rgb))
#
#   2. Keep Sigma's C exchange as the initialization:
#          RGB branch initially uses C_e
#          X/T branch initially uses C_rgb
#
#   3. Add delta-conditioned C readout coupling:
#          C_rgb_for_rgb = C_e   + lambda_rgb * R_rgb * (C_rgb - C_e)
#          C_e_for_e     = C_rgb + lambda_e   * R_e   * (C_e   - C_rgb)
#
#   4. lambda is initialized as 0, so the initial behavior is exactly:
#          your successful delta-CroMB + original C exchange
# =====================================================================


class Mlp_DCR(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.,
        channels_first=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Cross_Mamba_Attention_SSM_DCR(nn.Module):
    """
    DCR Cross Selective Scan Core.

    Compared with the cross-modal state core:
        - keeps original sigmoid delta modulation;
        - keeps C exchange;
        - adds a zero-init delta-conditioned C coupling.

    Input:
        x_rgb: B, L, d_inner
        x_e:   B, L, d_inner

    Output:
        y_rgb: B, L, d_inner
        y_e:   B, L, d_inner
    """

    def __init__(
        self,
        d_model=96,
        d_state=4,
        ssm_ratio=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        c_mix_scale=0.5,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # bound for C readout coupling strength
        self.c_mix_scale = c_mix_scale

        self.x_proj_1 = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.x_proj_2 = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )

        self.dt_proj_1 = self.dt_init(
            self.dt_rank,
            self.d_inner,
            dt_scale,
            dt_init,
            dt_min,
            dt_max,
            dt_init_floor,
            **factory_kwargs,
        )
        self.dt_proj_2 = self.dt_init(
            self.dt_rank,
            self.d_inner,
            dt_scale,
            dt_init,
            dt_min,
            dt_max,
            dt_init_floor,
            **factory_kwargs,
        )

        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)
        self.D_1 = self.D_init(self.d_inner)
        self.D_2 = self.D_init(self.d_inner)

        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)

        # -------------------------------------------------------------
        # 1) Your successful delta interaction.
        # Use bias=False to stay close to your first successful version.
        # -------------------------------------------------------------
        self.dt_gate_rgb = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.dt_gate_e = nn.Linear(self.d_inner, self.d_inner, bias=False)

        # -------------------------------------------------------------
        # 2) Delta-conditioned C readout coupling.
        # gate_rgb_bld is generated from x_e and modulates RGB branch;
        # gate_e_bld is generated from x_rgb and modulates X/T branch.
        #
        # c_gate_rgb maps d_inner -> d_state to match C shape.
        # -------------------------------------------------------------
        self.c_gate_rgb = nn.Linear(self.d_inner, self.d_state, bias=True)
        self.c_gate_e = nn.Linear(self.d_inner, self.d_state, bias=True)

        # zero-init coupling strength.
        # lambda = c_mix_scale * tanh(lambda_c)
        # initial lambda = 0, so C mixing is initially disabled.
        self.lambda_c_rgb = nn.Parameter(torch.zeros(1))
        self.lambda_c_e = nn.Parameter(torch.zeros(1))

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale

        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)

        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)

        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)

        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)

        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, L, _ = x_rgb.shape

        # B, L, D -> B, D, L
        x_rgb = x_rgb.permute(0, 2, 1).contiguous()
        x_e = x_e.permute(0, 2, 1).contiguous()

        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))

        dt_rgb, B_rgb, C_rgb = torch.split(
            x_dbl_rgb,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt_e, B_e, C_e = torch.split(
            x_dbl_e,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        # pre-softplus dt logits
        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()

        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", b=B, l=L).contiguous()
        dt_e = rearrange(dt_e, "d (b l) -> b d l", b=B, l=L).contiguous()

        # -------------------------------------------------------------
        # Successful delta interaction:
        #   RGB branch dt is modulated by X/T feature.
        #   X/T branch dt is modulated by RGB feature.
        # -------------------------------------------------------------
        x_e_bld = rearrange(x_e, "b d l -> b l d").contiguous()
        x_rgb_bld = rearrange(x_rgb, "b d l -> b l d").contiguous()

        gate_rgb_bld = torch.sigmoid(self.dt_gate_rgb(x_e_bld))
        gate_e_bld = torch.sigmoid(self.dt_gate_e(x_rgb_bld))

        gate_rgb = rearrange(gate_rgb_bld, "b l d -> b d l").contiguous()
        gate_e = rearrange(gate_e_bld, "b l d -> b d l").contiguous()

        dt_rgb = dt_rgb * gate_rgb
        dt_e = dt_e * gate_e

        A_rgb = -torch.exp(self.A_log_1.float())
        A_e = -torch.exp(self.A_log_2.float())

        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", b=B, l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", b=B, l=L).contiguous()
        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", b=B, l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", b=B, l=L).contiguous()

        # -------------------------------------------------------------
        # Delta-conditioned C readout coupling.
        #
        # Original successful version:
        #   RGB branch uses C_e
        #   X/T branch uses C_rgb
        #
        # DCR:
        #   RGB branch uses C_e + lambda * R * (C_rgb - C_e)
        #   X/T branch uses C_rgb + lambda * R * (C_e - C_rgb)
        #
        # Since lambda is zero at initialization,
        # this initially equals the successful hard C exchange.
        # -------------------------------------------------------------
        R_rgb = torch.sigmoid(self.c_gate_rgb(gate_rgb_bld))
        R_e = torch.sigmoid(self.c_gate_e(gate_e_bld))

        R_rgb = rearrange(R_rgb, "b l dstate -> b dstate l").contiguous()
        R_e = rearrange(R_e, "b l dstate -> b dstate l").contiguous()

        lambda_rgb = self.c_mix_scale * torch.tanh(self.lambda_c_rgb)
        lambda_e = self.c_mix_scale * torch.tanh(self.lambda_c_e)

        C_rgb_for_rgb_branch = C_e + lambda_rgb * R_rgb * (C_rgb - C_e)
        C_e_for_e_branch = C_rgb + lambda_e * R_e * (C_e - C_rgb)

        y_rgb = selective_scan(
            x_rgb,
            dt_rgb,
            A_rgb,
            B_rgb,
            C_rgb_for_rgb_branch,
            self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(),
            delta_softplus=True,
        )

        y_e = selective_scan(
            x_e,
            dt_e,
            A_e,
            B_e,
            C_e_for_e_branch,
            self.D_2.float(),
            delta_bias=self.dt_proj_2.bias.float(),
            delta_softplus=True,
        )

        y_rgb = rearrange(y_rgb, "b d l -> b l d").contiguous()
        y_rgb = self.out_norm_1(y_rgb)

        y_e = rearrange(y_e, "b d l -> b l d").contiguous()
        y_e = self.out_norm_2(y_e)

        return y_rgb, y_e


class CrossMambaFusion_SS2D_SSM_DCR(nn.Module):
    """
    Wrapper of DCR SSM core.
    Input / output shape follows Sigma CroMB:
        input:  B, H, W, C
        output: B, H, W, C
    """

    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        d_conv=3,
        conv_bias=True,
        dropout=0.,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        softmax_version=False,
        shared_ssm=False,
        c_mix_scale=0.5,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.CMA_ssm = Cross_Mamba_Attention_SSM_DCR(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            c_mix_scale=c_mix_scale,
            **kwargs,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        """
        x_rgb, x_e: B, H, W, C
        """
        B, H, W, _ = x_rgb.shape

        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)

        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()

            x_rgb_conv = self.act(self.conv2d(x_rgb_trans))
            x_e_conv = self.act(self.conv2d(x_e_trans))

            x_rgb_seq = rearrange(x_rgb_conv, "b d h w -> b (h w) d").contiguous()
            x_e_seq = rearrange(x_e_conv, "b d h w -> b (h w) d").contiguous()
        else:
            x_rgb_seq = rearrange(x_rgb, "b h w d -> b (h w) d").contiguous()
            x_e_seq = rearrange(x_e, "b h w d -> b (h w) d").contiguous()

        y_rgb, y_e = self.CMA_ssm(x_rgb_seq, x_e_seq)

        y_rgb = y_rgb.view(B, H, W, -1).contiguous()
        y_e = y_e.view(B, H, W, -1).contiguous()

        out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb))
        out_e = self.dropout_e(self.out_proj_e(y_e))

        return out_rgb, out_e


class CrossMambaFusionBlock_DCR(nn.Module):
    """
    DCR-CroMB block.

    This block only replaces CroMB.
    Original ConMB should still be used after this block.
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 4,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=0.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        c_mix_scale=0.5,
        **kwargs,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.op = CrossMambaFusion_SS2D_SSM_DCR(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            c_mix_scale=c_mix_scale,
            **kwargs,
        )

        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2_rgb = norm_layer(hidden_dim)
            self.norm2_e = norm_layer(hidden_dim)

            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp_rgb = Mlp_DCR(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                channels_first=False,
            )
            self.mlp_e = Mlp_DCR(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                channels_first=False,
            )

    def _forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb_cross, x_e_cross = self.op(x_rgb, x_e)

        x_rgb = x_rgb + self.drop_path1(x_rgb_cross)
        x_e = x_e + self.drop_path2(x_e_cross)

        if self.mlp_branch:
            x_rgb = x_rgb + self.drop_path1(self.mlp_rgb(self.norm2_rgb(x_rgb)))
            x_e = x_e + self.drop_path2(self.mlp_e(self.norm2_e(x_e)))

        return x_rgb, x_e

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_rgb, x_e)
        return self._forward(x_rgb, x_e)
