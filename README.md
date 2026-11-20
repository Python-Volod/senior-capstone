# Senior Capstone Project UTampa

Lung CT lesion segmentation research prototype. Current components include
NIfTI pair loading, expert annotation inspection, a small 3D U-Net, and a CPU
training/evaluation workflow. This is not a clinical diagnostic system.

- [Inspect an expert CT/mask pair](docs/inspect-pair.md)
- [Run the original one-patch mechanics demo](docs/unet-smoke-test.md)
- [Train, save checkpoints, and evaluate unseen patients](docs/training-evaluation.md)
- [Respond to professor feedback and plan model comparisons](docs/integration-plan.md)
- [One-patient training-fit diagnostic](docs/overfit-diagnostic.md)
- [Training-only hard-negative mining and crop variety](docs/hard-negative-training.md)

An initial real pair and a six-patient tumor-only pilot are available locally.
The pilot uses four training patients, one validation patient and one reserved
test patient; that is insufficient to establish generalizable accuracy.
Three saved real-data runs are completed locally: the BCE baseline, a
foreground-aware follow-up, and hard-negative training. The last reaches
validation Dice 0.008429 on one patient versus 0.003449 previously, with extensive
remaining false positives; useful generalization is not established.
See [pilot results](docs/pilot-training.md). `python -m model.predict --help`
describes checkpoint-based mask export without an expert annotation input.
Automated synthetic tests verify software behavior, not medical performance.
Keep scans, manifests, model weights, and generated reports in ignored local
directories. No teammate's 2D model is included in this branch.
