# Initialization weights

Place the ImageNet initialization files under `pretrained/vmamba/`:

- Tiny: [`vssmtiny_dp01_ckpt_epoch_292.pth`](https://drive.google.com/file/d/1W0EFQHvX4Cl6krsAwzlR-VKqQxfWEdM8/view)
- Small: [`vssmsmall_dp03_ckpt_epoch_238.pth`](https://drive.google.com/file/d/1671QXJ-faiNX4cYUlXxf8kCpAjeA4Oah/view)

The download sources are linked in [Sigma's training instructions](https://github.com/zifuwan/Sigma#training). They are used for training only; evaluation loads the full segmentation checkpoint.

Weight files are excluded from Git.
