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
| Random ops per image supply the diversity the priors lack | `dfkd/augment.py` |
| Frozen teacher labels each augmented view; student fits it | `dfkd/train.py`, `dfkd/distill.py` |
| Real data is reachable only for teacher training and evaluation | `dfkd/data.py` |

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
