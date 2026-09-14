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

## Teacher training and tuning

`dfkd train_teacher` runs **8 configurations**, including the configured baseline,
with at most 100 epochs each by default. Budget for up to 800 training epochs;
early stopping can reduce this. The reproducible random search samples distinct
combinations of learning rate, weight decay and label smoothing without replacement.
It uses the existing optimizer family. Edit `teacher_training.tuning.search_space`
for the architecture; these are starting ranges, not claimed optimal settings.

```bash
# Reproducible search, one fixed 90/10 stratified training/validation split
dfkd train_teacher --config configs/cifar10.yaml

# Single configuration, or a smaller search budget
dfkd train_teacher --config configs/mnist.yaml --set teacher_training.tuning.trials=1
dfkd train_teacher --config configs/cifar10.yaml --set teacher_training.tuning.trials=3

# --epochs now targets teacher epochs for train_teacher
dfkd train_teacher --config configs/cifar10.yaml --epochs 200
```

All trials use the same split, initialization seed and initial shuffle seed. Mild
training-only augmentation uses crop padding 4 on CIFAR and horizontal flips except
on MNIST. Set `teacher_training.augmentation.enabled=false` to disable it.
Validation is unaugmented. Split indices are saved relative to the original training
dataset. Seeded runs are reproducible on a fixed setup; CUDA kernels may still be
nondeterministic.

Checkpoint and trial selection minimize **validation cross-entropy (NLL)**.
Early stopping uses that same metric, with patience 20 and minimum improvement
0.0001; patience starts counting from epoch 60 to allow learning-rate decay.
Set `teacher_training.early_stopping.patience=0` to disable stopping. Increase
`min_epochs` when increasing the training budget or using a late LR drop.
The raw best checkpoint is saved even for improvements smaller than `min_delta`.
The official test set is evaluated **once, after all selection is complete**.
No full-training-set refit is performed: the exported model is the validation winner.

Each teacher run writes:

- `config.yaml`, `split.pt`, `trials.json`, `best_config.yaml`, and `results.json`.
- Per-trial resolved configs and summaries, plus flushed `metrics.csv` (scalars)
  and `metrics.jsonl` (including per-class metrics and confusion matrices).
- One `best.pt` for the winning trial, exported as the plain state dict at
  `teacher.checkpoint`, compatible with student distillation. An existing export
  is replaced after successful training; the timestamped run keeps its own copy.

Console output is limited to train/validation loss and accuracy each epoch, with
trial and epoch identifiers. Test loss and accuracy print once after model selection.

Every epoch saves train/validation NLL, accuracy, top-5 accuracy, macro precision,
recall and F1, weighted F1, balanced accuracy, 15-bin ECE, multiclass Brier score,
mean confidence, sample counts, training objective, LR, batch size, timing,
throughput, peak allocated CUDA memory, best epoch and stopping status.
JSONL also records per-class precision/recall/F1/support and confusion matrices
(rows are true classes, columns are predictions, in dataset class-index order).

Accuracies are percentages; precision/recall/F1, confidence and ECE are fractions.
Brier score is the unnormalized multiclass sum (range 0–2); NLL uses natural logs.
Macro metrics include all configured classes (undefined values are zero);
balanced accuracy averages recall over supported classes. Top-5 uses min(5, classes).
Training metrics describe the augmented minibatches seen during optimization;
`train_objective` includes label smoothing while `train_loss` is ordinary NLL.
Evaluation metrics describe a fixed checkpoint on unaugmented data.

No tuning framework or metrics dependency is added. Trial weights are not archived,
and metrics stream to disk rather than retaining predictions or epoch history.

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
