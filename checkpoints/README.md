# Trained model checkpoints

This initial release provides source code and training/evaluation instructions. No public checkpoint download is hosted at this stage.

Training outputs are written to `runs/<dataset>/<model>/checkpoint/`. Evaluate a saved checkpoint by passing its explicit path:

```bash
bash scripts/eval.sh mfnet /path/to/checkpoint.pth
```

Weight files are excluded from Git. Locally retained review weights are not part of the source archive.
