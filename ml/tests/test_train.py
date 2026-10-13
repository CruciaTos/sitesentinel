"""Tests for the synthetic data generator and the training pipeline (quality gate, champion/challenger)."""
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys

import numpy as np
import pytest

import generate_data
import ppe_model
import train

ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ML_DIR, "data", "ppe_pairs.csv")
TRAIN_PY = os.path.join(ML_DIR, "train.py")
VERSION_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}-[0-9a-f]{7}$")
GOOD_ROW = "hardhat,0.1,0.1,0.9,0.3,0.12,0.15,0.2,0.25,1,worn"


@pytest.fixture(autouse=True)
def no_gate_env(monkeypatch):
    """Tests rely on the documented defaults, whatever the CI environment sets."""
    monkeypatch.delenv("MODEL_MIN_ACCURACY", raising=False)
    monkeypatch.delenv("MODEL_MAX_REGRESSION", raising=False)


def run_train(tmp_path, *extra, data=DATA):
    out, report = tmp_path / "model.json", tmp_path / "report.json"
    code = train.main(["--data", str(data), "--out", str(out), "--report", str(report)] + list(extra))
    return code, out, report


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_generator_is_deterministic(tmp_path):
    assert generate_data.generate(500, seed=42) == generate_data.generate(500, seed=42)
    assert generate_data.generate(500, seed=42) != generate_data.generate(500, seed=7)
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    for path in (first, second):
        assert generate_data.main(["--rows", "300", "--seed", "42", "--out", str(path)]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_generator_schema(tmp_path):
    out = tmp_path / "pairs.csv"
    assert generate_data.main(["--rows", "1000", "--out", str(out)]) == 0
    assert b"\r\n" not in out.read_bytes()
    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == generate_data.COLUMNS == train.COLUMNS
    assert len(rows) == 1000
    for item_type in ppe_model.ITEM_TYPES:
        labels = [int(r["label"]) for r in rows if r["item_type"] == item_type]
        assert len(labels) == 500
        assert 0.4 < sum(labels) / len(labels) < 0.6
    scenarios = {r["scenario"].split("+")[0] for r in rows}
    assert scenarios == {"worn", "held", "ground", "neighbour", "far"}
    for r in rows:
        assert r["label"] in ("0", "1")
        values = [r[c] for c in train.PERSON_COLUMNS + train.ITEM_COLUMNS]
        assert all(len(v.split(".")[1]) <= 4 for v in values)
        p_ymin, p_xmin, p_ymax, p_xmax, i_ymin, i_xmin, i_ymax, i_xmax = map(float, values)
        assert 0.0 <= p_ymin < p_ymax <= 1.0 and 0.0 <= p_xmin < p_xmax <= 1.0
        assert 0.0 <= i_ymin < i_ymax <= 1.0 and 0.0 <= i_xmin < i_xmax <= 1.0


def test_features_shape():
    rows, _ = train.read_data(DATA)
    for item_type in ppe_model.ITEM_TYPES:
        x, y = train.xy(rows, item_type)
        assert np.asarray(x).shape == (len(y), len(ppe_model.FEATURES))
        assert set(y) == {0, 1}


def test_crlf_checkout_has_the_same_data_hash(tmp_path):
    with open(DATA, "rb") as f:
        lf = f.read().replace(b"\r\n", b"\n")
    (tmp_path / "lf.csv").write_bytes(lf)
    (tmp_path / "crlf.csv").write_bytes(lf.replace(b"\n", b"\r\n"))
    rows_lf, sha_lf = train.read_data(str(tmp_path / "lf.csv"))
    rows_crlf, sha_crlf = train.read_data(str(tmp_path / "crlf.csv"))
    assert sha_lf == sha_crlf == hashlib.sha256(lf).hexdigest()
    assert len(rows_lf) == len(rows_crlf)


def test_training_on_committed_data_passes_the_gate(tmp_path):
    summary = tmp_path / "summary.md"
    code, out, report = run_train(tmp_path, "--summary", str(summary))
    assert code == 0
    spec = read_json(out)
    model = ppe_model.LogisticModel(spec)
    assert VERSION_RE.match(model.version)
    assert spec["version"].endswith("-" + spec["data_sha256"][:7])
    assert spec["trained_at"].endswith("Z")
    assert spec["features"] == ppe_model.FEATURES
    assert spec["training"]["seed"] == 42
    for item_type in ppe_model.ITEM_TYPES:
        assert spec["metrics"][item_type]["accuracy"] >= 0.9
        params = spec["models"][item_type]
        assert all(round(v, 6) == v for v in params["mean"] + params["scale"] + params["coef"])

    result = read_json(report)
    assert result["gate"]["passed"] and result["gate"]["reasons"] == []
    assert result["model_written"] == str(out)
    assert result["previous_model"] is None
    assert result["candidate"]["metrics"] == spec["metrics"]
    assert result["baseline_rule"]["version"] == "rules-0.6"
    assert result["candidate"]["metrics"]["overall"]["f1"] > result["baseline_rule"]["metrics"]["overall"]["f1"]
    confusion = result["candidate"]["confusion_matrices"]
    assert sum(sum(c.values()) for c in confusion.values()) == result["data"]["test_rows"]

    text = summary.read_text(encoding="utf-8")
    assert "## PPE model training" in text and "| hardhat |" in text and "Quality gate: PASSED" in text


def test_retraining_on_the_same_data_passes_champion_challenger(tmp_path):
    assert run_train(tmp_path)[0] == 0
    first = read_json(tmp_path / "model.json")
    code, out, report = run_train(tmp_path)
    assert code == 0
    result = read_json(report)
    assert result["previous_model"]["version"] == first["version"]
    assert result["previous_model"]["metrics"] == result["candidate"]["metrics"]
    assert read_json(out)["models"] == first["models"]


@pytest.mark.parametrize("use_env", [False, True])
def test_gate_fails_below_min_accuracy(tmp_path, monkeypatch, use_env):
    if use_env:
        monkeypatch.setenv("MODEL_MIN_ACCURACY", "1.01")
        code, out, report = run_train(tmp_path)
    else:
        code, out, report = run_train(tmp_path, "--min-accuracy", "1.01")
    assert code == 1
    assert not out.exists()
    result = read_json(report)
    assert not result["gate"]["passed"]
    assert result["model_written"] is None
    assert any("below the minimum" in reason for reason in result["gate"]["reasons"])


def test_champion_challenger_blocks_a_worse_model(tmp_path):
    assert run_train(tmp_path)[0] == 0
    champion = (tmp_path / "model.json").read_bytes()

    # Challenger data: most worn items mislabelled as not worn, so the new model learns to say "not worn".
    with open(DATA, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rng = random.Random(1)
    for row in rows[1:]:
        if row[9] == "1" and rng.random() < 0.7:
            row[9] = "0"
    bad = tmp_path / "bad.csv"
    with open(bad, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)

    code, out, report = run_train(tmp_path, "--min-accuracy", "0", data=bad)
    assert code == 1
    assert out.read_bytes() == champion
    result = read_json(report)
    assert result["previous_model"]["version"] == read_json(out)["version"]
    assert any("overall F1" in reason for reason in result["gate"]["reasons"])


def test_unusable_current_model_is_replaced(tmp_path):
    (tmp_path / "model.json").write_text("{not json", encoding="utf-8")
    code, out, report = run_train(tmp_path)
    assert code == 0
    assert ppe_model.LogisticModel(read_json(out)).algorithm == "logistic_regression"
    assert read_json(report)["previous_model"] is None


@pytest.mark.parametrize("content", [
    "a,b\n1,2\n",
    ",".join(train.COLUMNS) + "\n" + GOOD_ROW.replace("hardhat", "gloves") + "\n",
    ",".join(train.COLUMNS) + "\n" + GOOD_ROW.replace(",1,worn", ",7,worn") + "\n",
    ",".join(train.COLUMNS) + "\n" + GOOD_ROW.replace("0.12", "abc") + "\n",
    ",".join(train.COLUMNS) + "\n" + GOOD_ROW + "\n",
])
def test_bad_input_exits_2(tmp_path, content):
    data = tmp_path / "data.csv"
    data.write_text(content, encoding="utf-8")
    code, out, report = run_train(tmp_path, data=data)
    assert code == 2
    assert not out.exists() and not report.exists()


def test_missing_data_file_exits_2(tmp_path):
    assert run_train(tmp_path, data=tmp_path / "missing.csv")[0] == 2


@pytest.mark.parametrize("extra, expected", [([], 0), (["--min-accuracy", "1.01"], 1), (["--data", "nope.csv"], 2)])
def test_cli_exit_codes_from_any_directory(tmp_path, extra, expected):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MODEL_")}
    args = [sys.executable, TRAIN_PY, "--out", str(tmp_path / "m.json"), "--report", str(tmp_path / "r.json")]
    result = subprocess.run(args + extra, cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stdout + result.stderr
    assert "[MLOps]" in result.stdout
