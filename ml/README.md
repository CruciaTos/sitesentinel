# PPE association model (MLOps)

The detector returns separate boxes for people, hard hats and vests. This model decides, for each
(person, item) pair, whether that person is **wearing** the item. It replaces the old hand-written
rule ("most of the item's box lies inside the person's box"), which counts a hat held at the waist,
or a vest worn by someone standing in front, as worn.

- Features: `features()` in `backend/ppe_model.py` (item overlap, IoU, position relative to the
  person, size relative to the person). Training and serving call the same function.
- Model: one standardized logistic regression per item type (scikit-learn), exported as plain JSON
  to `backend/model/ppe_model.json`. The backend scores it in pure Python, so the app image needs
  no numpy or scikit-learn. If the file is missing or broken, the backend falls back to the old rule.
- Data: `ml/data/ppe_pairs.csv`, 6000 synthetic labelled pairs from `generate_data.py` (seed 42):
  worn items plus hard negatives (held at the waist, on the ground, worn by a neighbour or by
  someone on a scaffold above, far away), cropped people, odd box sizes and ~3% label noise.

| File | Purpose |
|---|---|
| `generate_data.py` | Writes the synthetic dataset (deterministic for a given `--seed`) |
| `train.py` | Trains, evaluates, applies the quality gate, writes the model and report |
| `reports/metrics.json` | Report of the last run (the committed one describes the baseline model) |
| `tests/test_train.py` | Data, training, gate and champion/challenger tests |

## Retrain

```sh
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r ml/requirements.txt
python ml/generate_data.py      # only to rebuild the synthetic data
python ml/train.py              # writes backend/model/ppe_model.json if the gate passes
```

`train.py` options: `--data`, `--out`, `--report`, `--summary FILE` (appends a Markdown summary,
e.g. `$GITHUB_STEP_SUMMARY`), `--min-accuracy`, `--max-regression`, `--seed`. Relative paths are
taken from the repo root, so it runs from any directory.

Tests: `pip install pytest`, then `cd ml && pytest`. Lint: `cd ml && flake8 .`

## Quality gate

| Check | Default | Override |
|---|---|---|
| Test accuracy of every item type | >= 0.90 | `--min-accuracy` / `MODEL_MIN_ACCURACY` |
| Champion/challenger: overall F1 vs the model currently at `--out` | at most 0.02 lower | `--max-regression` / `MODEL_MAX_REGRESSION` |

Both models are scored on the same held-out 20% split (stratified, seed 42), through the same code
the backend serves with. If a check fails the model file is left untouched and the script exits 1,
which fails the CI stage. Exit codes: `0` model written, `1` gate failed, `2` bad input data.
The report (`--report`) is written either way, with metrics, confusion matrices, accuracy per
scenario, the overlap rule baseline, the previous model's metrics and the gate verdict.

## Versioning

- Model version = `YYYY.MM.DD-<first 7 hex of the data's sha256>` (UTC date), e.g. `2026.10.13-6d1b6d4`.
  The file also records `data_sha256`, `trained_at`, the seed and the scikit-learn version.
- Data, model and report are committed to Git, so every app commit pins one model.
- The running backend reports the model version at `/healthz`, `/api/version` and in the
  `sitesentinel_model_info` Prometheus metric.

## Retraining triggers

- **GitHub Actions** (`.github/workflows/ci-cd.yml`, job `train-model`): runs on every push to `main`
  (for example a change to `ml/data/`), on pull requests, on the weekly schedule, and on demand with
  "Run workflow" (workflow_dispatch). It uploads the model and report as the `ppe-model` artifact,
  and the image build uses that model.
- **Jenkins** (`Jenkinsfile`, stage "Train & Validate Model"): runs the same `train.py` in a venv.
  A failed gate stops the pipeline before the image is built.

## Baseline results (`reports/metrics.json`)

| Item | Accuracy | F1 | Overlap rule accuracy |
|---|---|---|---|
| Hard hat | 0.9733 | 0.9736 | 0.8167 |
| Vest | 0.9650 | 0.9652 | 0.8267 |
| Overall | 0.9692 | 0.9694 | 0.8217 |
