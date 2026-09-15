# SNKD

Data-free knowledge distillation from random synthetic images. A frozen teacher
provides soft labels for an augmented mixture of uniform, Gaussian, gradient,
Perlin, Gabor and checkerboard priors. The student never trains on real images.

## Install

Requires Python 3.10+ and PyTorch; CUDA is used automatically when available.

```bash
git clone https://github.com/dexterdev/SNKD.git
cd SNKD
pip install -e '.[dev]'
```

## Model configurations

| Dataset | Teacher → student | Config |
| --- | --- | --- |
| MNIST | LeNet-5 → LeNet-5/2 | `configs/mnist.yaml` |
| Fashion-MNIST | LeNet-5 → LeNet-5/2 | `configs/fashionmnist.yaml` |
| CIFAR-10 | ResNet-34 → ResNet-18 | `configs/cifar10.yaml` |
| CIFAR-100 | ResNet-34 → ResNet-18 | `configs/cifar100.yaml` |
| CIFAR-10 | ViT-8 → ViT-4 | `configs/cifar10_vit.yaml` |
| CIFAR-100 | ViT-8 → ViT-4 | `configs/cifar100_vit.yaml` |

ViT uses the [pinned small-dataset architecture](https://github.com/lucidrains/vit-pytorch/blob/e1b08c15b9b237329d30324ce40579d4d4afc761/vit_pytorch/vit_for_small_dataset.py)
with SPT and LSA. Teacher/student depths are 8/4; both default to patch size 4,
dimension 256, 4 heads, MLP dimension 1024 and dropout 0.1.

## Run

Choose a config from the table and use it throughout:

```bash
CONFIG=configs/cifar10_vit.yaml
dfkd inspect --config "$CONFIG"
dfkd train_teacher --config "$CONFIG"
dfkd distill --config "$CONFIG"
dfkd evaluate --config "$CONFIG" --checkpoint runs/<student-run>/best.pt
```

Replace `runs/<student-run>/best.pt` with the actual student checkpoint path.
Datasets download to `data/` by default. Distillation requires the teacher
checkpoint at `teacher.checkpoint`; teacher training writes it automatically.

## Training settings

Defaults are in `configs/base.yaml`; dataset configs override them.
Use `--set dotted.key=value` for overrides:

```bash
dfkd train_teacher --config "$CONFIG" --epochs 200 --set teacher_training.optimizer.lr=0.05
dfkd distill --config "$CONFIG" --temperature 4 --set optimizer.lr=0.01
```

- **Teacher:** one run, up to 100 epochs by default, with a seeded 90/10
  training/validation split. Validation loss selects the best checkpoint and
  controls early stopping (patience 20, minimum improvement 0.0001; patience
  starts counting at epoch 60). Set `teacher_training.early_stopping.patience=0`
  to disable stopping. Train/validation/test loss and accuracy print each epoch;
  test metrics are reporting-only.
- **Student:** 200 epochs on 200,000 synthetic samples by default. Batch sizes
  progress through `[16, 32, 64, 128, 256, 512, 1024, 2048]`, with cosine LR
  restarting in each phase. Change `training.batch_size_schedule` or
  `scheduler.phase_resets` as needed. Augmentation defaults to crop/flip plus
  3 of 17 random operations; MNIST disables horizontal flips.

`--epochs` controls the selected training command. Architecture options are under
`teacher.kwargs` and `student.kwargs`; supplied hyperparameters are starting defaults.

## Outputs

Runs are saved under `runs/<name>_<timestamp>/`.

| Run | Files |
| --- | --- |
| Teacher | `config.yaml`, `split.pt`, `metrics.csv`, `metrics.jsonl`, `results.json`, `best.pt`; best weights also exported to `teacher.checkpoint` |
| Student | `config.yaml`, `metrics.csv`, `results.json`, `best.pt`, `last.pt` |

Teacher files retain detailed classification, calibration and timing metrics;
JSONL includes per-class scores and confusion matrices. Accuracy is in percent;
precision/recall/F1 and calibration error are fractions.

## Tests and license

`pytest -q` runs CPU tests using artificial data, without dataset downloads.

MIT. The vendored ViT source retains Phil Wang's MIT license.
