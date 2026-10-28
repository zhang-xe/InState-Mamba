# Third-party notices

InState-Mamba incorporates and adapts code from the projects below. Existing copyright and license headers remain in the source files. These components retain their original license terms, including in a distribution of InState-Mamba.

| Project | Attribution | License text |
| --- | --- | --- |
| [Sigma](https://github.com/zifuwan/Sigma) | Copyright (c) 2024 Zifu Wan | [MIT](docs/Sigma-MIT.txt) |
| [VMamba](https://github.com/MzeroMiko/VMamba) | Copyright (c) 2024 MzeroMiko | [MIT](docs/VMamba-MIT.txt) |
| [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) | Copyright (c) 2022 Huayao Liu | [MIT](docs/CMX-MIT.txt) |
| [Mamba](https://github.com/state-spaces/mamba) | Albert Gu, Tri Dao and contributors | [Apache-2.0](docs/Mamba-Apache-2.0.txt) |
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | Microsoft Corporation; `dual_swin.py` retains its 2021 copyright and author header | [MIT](docs/Swin-MIT.txt) |

Selective-scan source also retains its file-level attribution to Albert Gu, Tri Dao and VMamba contributors. Mamba-derived portions remain subject to Apache-2.0. The NVIDIA copyright, redistribution conditions and disclaimer in [uninitialized_copy.cuh](models/encoders/selective_scan/csrc/selective_scan/uninitialized_copy.cuh) are retained in full and continue to apply to that file. Attribution links in `static_switch.h` are also retained.

## InState-Mamba contributions

Copyright (c) 2026 Xinwei Zhang. Original InState-Mamba code contributions and documentation are provided under the [MIT License](LICENSE), subject to the applicable third-party terms above.

InState-Mamba modifications include WRCSI, BWM and RCSF integration, module switches, and release configuration and entry-point preparation. The selective-scan build script includes a CUDA toolkit availability check; its modification notice is recorded in the file header. This release also identifies the applicable license families in the extension's package metadata. The CUDA kernel implementation is unchanged by this release preparation.

## Figures, datasets and weights

The overview figure has a [separate copyright and permission notice](assets/README.md) and is excluded from the code's MIT license.

Dataset downloads and pretrained weights are governed by their original providers' terms. The links in [README.md](README.md) do not grant a new license to those resources. Dataset archives and model weights are not bundled with this source release.
