# DFKD from random synthetic priors

Data-free knowledge distillation without a generator. The student never sees a real
training image: its inputs are drawn from parametric random priors (uniform, Gaussian,
linear gradients, Perlin noise, Gabor patches, checkerboards), heavily augmented, and
labelled by a frozen teacher's soft logits. Real data is used only to train the teacher
and to measure held-out accuracy.

## Install

```bash
pip install -e '.[dev]'
```

## Use

```bash
dfkd inspect       --config configs/mnist.yaml          # resolved config + parameter counts
dfkd train_teacher --config configs/mnist.yaml          # the only step that reads real train data
dfkd distill       --config configs/mnist.yaml          # student training on synthetic data
dfkd evaluate      --config configs/mnist.yaml --checkpoint runs/<run>/best.pt
```

Configs: `mnist.yaml` and `fashionmnist.yaml` (LeNet-5 → LeNet-5/2),
`cifar10.yaml` and `cifar100.yaml` (ResNet-34 → ResNet-18).

Override anything from the command line:

```bash
dfkd distill --config configs/cifar10.yaml --temperature 4 --epochs 50 \
  --set synthetic_data.priors.gabor.weight=3 --set optimizer.lr=0.05
```

Each distillation writes `runs/<name>_<timestamp>/` containing the resolved
`config.yaml`, `metrics.csv`, `results.json`, and `best.pt` / `last.pt`.

## How it works

| Step | Module |
| --- | --- |
| Sample `i` is a pure function of `(seed, i)` — no corpus is stored | `dfkd/synthetic.py` |
| Query augmentation supplies the diversity the priors lack | `dfkd/augment.py` |
| Batch size and learning rate advance together | `dfkd/schedule.py` |
| Frozen teacher labels each augmented view; student fits it | `dfkd/train.py`, `dfkd/distill.py` |
| Real data is reachable only for teacher training and evaluation | `dfkd/data.py` |

### Query augmentation

Each synthetic image is augmented independently, in two sequential stages:

1. **Geometric (2 transforms, always applied):** `RandomCrop(padding=4, reflect)` then
   `RandomHorizontalFlip(p=0.5)`.
2. **RandAug-style (`n` of 17, without replacement):** `num_random_ops` operations are
   drawn from the 17-operation pool as a combination — no operation is drawn twice —
   and applied in the order drawn. Default `n = 3`.

The pool: rotation, affine, perspective, random_resized_crop, color_jitter, grayscale,
gaussian_blur, sharpness, autocontrast, histogram_equalization, posterization,
solarization, inversion, gaussian_noise, salt_and_pepper_noise, cutout,
random_color_region_erasing.

**MNIST runs with the flip disabled** — a mirrored digit is a different digit — so it
uses one geometric transform rather than two. Fashion-MNIST and the CIFAR configs keep
the flip.

### Batch-size and LR scheduling

Student training splits its epochs into one contiguous phase per entry of
`training.batch_size_schedule` (earlier phases absorb the remainder). The batch size
steps up at each phase boundary while the LR follows its own curve:

```yaml
training:
  epochs: 200
  batch_size_schedule: [16, 32, 64, 128, 256, 512, 1024, 2048]
scheduler:
  name: cosine        # constant | cosine | step | warmup_cosine
  phase_resets: true
  min_lr_ratio: 0.0
```

With `phase_resets: true` the cosine restarts inside each phase, so all eight
batch-size regimes train under a full decay (a warm restart) — 25 epochs at batch 16
with LR 0.01 → 0, then 25 at batch 32 with LR 0.01 → 0, and so on. Set it to `false`
for a single global decay across all 200 epochs, which leaves the large-batch phases
training at a near-zero LR. `metrics.csv` records the phase and batch size per epoch.

## Configuration

`configs/base.yaml` holds the defaults; a dataset config `extends` it and overrides the
dataset, architectures and teacher checkpoint path. Priors, augmentation operations,
losses, models and datasets are registries — add an entry and reference it by name
instead of branching in the training loop.

## Tests

```bash
pytest -q
```

The suite runs on CPU in under a minute and never downloads a dataset.

## License

MIT, see `LICENSE`.
