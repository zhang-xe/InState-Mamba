def build_checkpoint_backbone(
    experiment,
    backbone,
    use_wi=True,
    use_ri=True,
    use_bwm=None,
    use_rcsf=True,
    use_rcsf_feat=True,
    use_rcsf_diff=True,
    use_rcsf_coact=True,
    bwm_mode='full',
    *,
    use_hbwr=None,
):
    # Keep the fifth positional switch and old keyword compatible.
    if use_bwm is not None and use_hbwr is not None:
        if bool(use_bwm) != bool(use_hbwr):
            raise ValueError('Conflicting use_bwm/use_hbwr switches.')
    enabled = use_bwm if use_bwm is not None else use_hbwr
    use_bwm = True if enabled is None else bool(enabled)
    if bwm_mode not in ('full', 'shared_directions', 'shared_channels'):
        raise ValueError('Invalid bwm_mode: ' + repr(bwm_mode))
    if experiment != 'ablation2' and bwm_mode != 'full':
        raise ValueError('Select ablation2 for BWM sharing experiments.')

    if experiment == 'legacy':
        legacy_names = {
            'dwr_tiny': 'dwr_tiny',
            'dwr_small': 'dwr_small',
            'dwr_base': 'dwr_base',
        }
        return legacy_names.get(backbone, backbone)

    if experiment == 'ablation1':
        names = []
        if use_wi:
            names.append('WI')
        if use_ri:
            names.append('RI')
        if use_bwm:
            names.append('BWM')
        if use_rcsf:
            names.append('RCSF')
        empty_suffix = 'baseline'
    elif experiment == 'ablation2':
        if backbone not in ('dwr_tiny', 'dwr_small', 'dwr_base'):
            raise ValueError('BWM ablation2 requires a dwr_tiny/small/base backbone.')
        if not use_wi or not use_ri or use_rcsf:
            raise ValueError('BWM ablation2 requires WI=True, RI=True, RCSF=False.')
        if not use_bwm:
            if bwm_mode != 'full':
                raise ValueError('For the WI+RI baseline set use_bwm=False and bwm_mode=full.')
            return f'ablation2_{backbone}_WI_RI_without_BWM'
        return f'ablation2_{backbone}_WI_RI_BWM_{bwm_mode}'
    else:
        raise ValueError(
            "experiment must be 'legacy', 'ablation1', or 'ablation2'"
        )

    suffix = '_'.join(names) if names else empty_suffix
    return f'{experiment}_{suffix}'
