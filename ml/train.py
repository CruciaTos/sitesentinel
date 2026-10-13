"""Train the PPE association model, check it against the quality gate and export it for the backend.

Reads labelled (person box, item box) pairs, fits one standardized logistic regression per item
type on ppe_model.features() and evaluates it on a held-out split, next to the hand-written overlap
rule and the model currently at --out. The new model is written only if every item type reaches
--min-accuracy and its overall F1 is at most --max-regression below the current model's
(champion/challenger). The report (and the optional Markdown summary) is written either way.

Exit codes: 0 model written, 1 quality gate failed, 2 bad input.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from datetime import datetime, timezone

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
import ppe_model  # noqa: E402

PERSON_COLUMNS = ["p_ymin", "p_xmin", "p_ymax", "p_xmax"]
ITEM_COLUMNS = ["i_ymin", "i_xmin", "i_ymax", "i_xmax"]
COLUMNS = ["item_type"] + PERSON_COLUMNS + ITEM_COLUMNS + ["label", "scenario"]
TEST_SIZE = 0.2
MIN_ROWS_PER_CLASS = 5


class DataError(Exception):
    """The training data cannot be used (exit code 2)."""


def log(message):
    print("[MLOps] " + message, flush=True)


def data_sha256(raw):
    """Hash of the file bytes with CRLF turned into LF, so Windows and Linux checkouts agree."""
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def read_data(path):
    """(rows, sha256) where each row has item_type, person, item, label, scenario and features x."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as err:
        raise DataError("cannot read {}: {}".format(path, err.strerror or err))
    try:
        text = raw.replace(b"\r\n", b"\n").decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DataError("{} is not UTF-8 text".format(path))
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise DataError("{} is missing columns: {}".format(path, ", ".join(missing)))
    rows = []
    for rec in reader:
        try:
            item_type = rec["item_type"]
            if item_type not in ppe_model.ITEM_TYPES:
                raise ValueError("unknown item_type {!r}".format(item_type))
            label = int(rec["label"])
            if label not in (0, 1):
                raise ValueError("label must be 0 or 1, got {}".format(label))
            person = [float(rec[c]) for c in PERSON_COLUMNS]
            item = [float(rec[c]) for c in ITEM_COLUMNS]
            if not all(math.isfinite(v) for v in person + item):
                raise ValueError("box coordinates must be finite numbers")
        except (TypeError, ValueError) as err:
            raise DataError("{} line {}: {}".format(path, reader.line_num, err))
        rows.append({
            "item_type": item_type,
            "person": person,
            "item": item,
            "label": label,
            "scenario": (rec["scenario"] or "").split("+")[0],
            "x": ppe_model.features(person, item, item_type),
        })
    for item_type in ppe_model.ITEM_TYPES:
        labels = [r["label"] for r in rows if r["item_type"] == item_type]
        if min(labels.count(0), labels.count(1)) < MIN_ROWS_PER_CLASS:
            raise DataError("{} needs at least {} worn and {} not-worn {} rows".format(
                path, MIN_ROWS_PER_CLASS, MIN_ROWS_PER_CLASS, item_type))
    return rows, data_sha256(raw)


def split(rows, seed):
    """Stratified (by item type and label) train/test split."""
    strata = ["{}:{}".format(r["item_type"], r["label"]) for r in rows]
    train_idx, test_idx = train_test_split(
        list(range(len(rows))), test_size=TEST_SIZE, random_state=seed, stratify=strata)
    return [rows[i] for i in sorted(train_idx)], [rows[i] for i in sorted(test_idx)]


def xy(rows, item_type):
    """Feature matrix and labels for one item type."""
    part = [r for r in rows if r["item_type"] == item_type]
    return [r["x"] for r in part], [r["label"] for r in part]


def _rounded(values, digits=6):
    return [round(float(v), digits) for v in values]


def fit(train_rows, seed):
    """Per item type: the scaler and logistic regression parameters, as stored in the model file."""
    models = {}
    for item_type in ppe_model.ITEM_TYPES:
        x, y = xy(train_rows, item_type)
        scaler = StandardScaler().fit(x)
        clf = LogisticRegression(max_iter=1000, random_state=seed).fit(scaler.transform(x), y)
        models[item_type] = {
            "mean": _rounded(scaler.mean_),
            "scale": _rounded(scaler.scale_),
            "coef": _rounded(clf.coef_[0]),
            "intercept": round(float(clf.intercept_[0]), 6),
        }
    return models


def _scores(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "accuracy": round((tp + tn) / len(y_true), 4) if y_true else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    return metrics, {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def evaluate(model, rows):
    """Metrics of a ppe_model model object (the serving code path) on rows."""
    result = {"version": model.version, "algorithm": model.algorithm,
              "metrics": {}, "confusion_matrices": {}, "accuracy_by_scenario": {}}
    all_true, all_pred, hits = [], [], {}
    for item_type in ppe_model.ITEM_TYPES:
        part = [r for r in rows if r["item_type"] == item_type]
        y_true = [r["label"] for r in part]
        y_pred = [int(model.wears(r["person"], r["item"], item_type)) for r in part]
        metrics, confusion = _scores(y_true, y_pred)
        result["metrics"][item_type] = metrics
        result["confusion_matrices"][item_type] = confusion
        all_true += y_true
        all_pred += y_pred
        for r, p in zip(part, y_pred):
            total, right = hits.get(r["scenario"], (0, 0))
            hits[r["scenario"]] = (total + 1, right + int(p == r["label"]))
    overall, _ = _scores(all_true, all_pred)
    result["metrics"]["overall"] = {"accuracy": overall["accuracy"], "f1": overall["f1"]}
    result["accuracy_by_scenario"] = {
        name: round(right / total, 4) for name, (total, right) in sorted(hits.items())}
    return result


def load_previous(path):
    """The current (champion) model at path, or None if there is no usable one."""
    if not os.path.exists(path):
        log("No current model at {}; skipping champion/challenger".format(_show(path)))
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return ppe_model.LogisticModel(json.load(f))
    except (OSError, ValueError, KeyError, TypeError) as err:
        log("Current model at {} is unusable ({}); skipping champion/challenger".format(_show(path), err))
        return None


def check_gate(candidate, previous, min_accuracy, max_regression):
    """List of reasons the candidate must not replace the current model (empty = pass)."""
    reasons = []
    for item_type in ppe_model.ITEM_TYPES:
        accuracy = candidate["metrics"][item_type]["accuracy"]
        if accuracy < min_accuracy:
            reasons.append("{} accuracy {:.4f} is below the minimum {:.4f}".format(
                item_type, accuracy, min_accuracy))
    if previous is not None:
        new_f1 = candidate["metrics"]["overall"]["f1"]
        old_f1 = previous["metrics"]["overall"]["f1"]
        if new_f1 < old_f1 - max_regression:
            reasons.append("overall F1 {:.4f} is more than {:.4f} below the current model {} ({:.4f})".format(
                new_f1, max_regression, previous["version"], old_f1))
    return reasons


def _show(path):
    """Repo-relative path with forward slashes when inside the repo, else the path as given."""
    full = os.path.abspath(path)
    try:
        rel = os.path.relpath(full, ROOT)
    except ValueError:  # another drive on Windows
        return full
    return full if rel.startswith("..") else rel.replace(os.sep, "/")


def _write_json(path, data):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _summary(report):
    data, cand, rule, prev, gate = (report["data"], report["candidate"], report["baseline_rule"],
                                    report["previous_model"], report["gate"])
    lines = [
        "## PPE model training",
        "",
        "- Data: `{}`, {} rows, sha256 `{}`".format(data["path"], data["rows"], data["sha256"][:12]),
        "- Split: {} train / {} test (stratified, seed {})".format(
            data["train_rows"], data["test_rows"], data["seed"]),
        "- Candidate: `{}` ({}, scikit-learn {})".format(cand["version"], cand["algorithm"], sklearn.__version__),
        "",
        "| Item | Accuracy | Precision | Recall | F1 | Overlap rule accuracy | Overlap rule F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for item_type in ppe_model.ITEM_TYPES:
        m, r = cand["metrics"][item_type], rule["metrics"][item_type]
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            item_type, m["accuracy"], m["precision"], m["recall"], m["f1"], r["accuracy"], r["f1"]))
    m, r = cand["metrics"]["overall"], rule["metrics"]["overall"]
    lines.append("| **overall** | {:.4f} | | | {:.4f} | {:.4f} | {:.4f} |".format(
        m["accuracy"], m["f1"], r["accuracy"], r["f1"]))
    lines += ["", "| Scenario | Model accuracy | Overlap rule accuracy |", "|---|---|---|"]
    for name, accuracy in cand["accuracy_by_scenario"].items():
        lines.append("| {} | {:.4f} | {:.4f} |".format(name, accuracy, rule["accuracy_by_scenario"][name]))
    lines.append("")
    if prev:
        delta = cand["metrics"]["overall"]["f1"] - prev["metrics"]["overall"]["f1"]
        lines.append("Champion/challenger: current model `{}` overall F1 {:.4f}, candidate {:.4f} "
                     "({:+.4f}, allowed regression {:.4f}).".format(
                         prev["version"], prev["metrics"]["overall"]["f1"],
                         cand["metrics"]["overall"]["f1"], delta, gate["max_regression"]))
    else:
        lines.append("Champion/challenger: no current model to compare with.")
    lines.append("")
    if gate["passed"]:
        lines.append("**Quality gate: PASSED** (every item accuracy >= {:.2f}). Model written to `{}`.".format(
            gate["min_accuracy"], report["model_written"]))
    else:
        lines.append("**Quality gate: FAILED**, model not written:")
        lines.append("")
        lines += ["- " + reason for reason in gate["reasons"]]
    return "\n".join(lines) + "\n\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the PPE association model behind a quality gate.",
        epilog="Relative paths are taken from the repo root, so this works from any directory.")
    parser.add_argument("--data", default="ml/data/ppe_pairs.csv", help="labelled pairs CSV")
    parser.add_argument("--out", default="backend/model/ppe_model.json",
                        help="model JSON to write; the model already there is the champion")
    parser.add_argument("--report", default="ml/reports/metrics.json", help="metrics report JSON to write")
    parser.add_argument("--summary", help="Markdown file to append a summary to (e.g. $GITHUB_STEP_SUMMARY)")
    parser.add_argument("--min-accuracy", type=float, default=os.environ.get("MODEL_MIN_ACCURACY", "0.90"),
                        help="minimum test accuracy per item type (env MODEL_MIN_ACCURACY, default 0.90)")
    parser.add_argument("--max-regression", type=float, default=os.environ.get("MODEL_MAX_REGRESSION", "0.02"),
                        help="largest allowed overall F1 drop vs the current model "
                             "(env MODEL_MAX_REGRESSION, default 0.02)")
    parser.add_argument("--seed", type=int, default=42, help="seed for the split and the solver (default 42)")
    args = parser.parse_args(argv)
    data_path, out_path, report_path = (os.path.join(ROOT, p) for p in (args.data, args.out, args.report))

    try:
        rows, sha = read_data(data_path)
        train_rows, test_rows = split(rows, args.seed)
    except DataError as err:
        log("ERROR: " + str(err))
        return 2
    log("Loaded {} rows from {} (sha256 {})".format(len(rows), _show(data_path), sha[:12]))
    log("Split {} train / {} test (stratified by item type and label, seed {})".format(
        len(train_rows), len(test_rows), args.seed))

    now = datetime.now(timezone.utc)
    spec = {
        "name": "ppe-association",
        "version": "{}-{}".format(now.strftime("%Y.%m.%d"), sha[:7]),
        "algorithm": "logistic_regression",
        "trained_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_sha256": sha,
        "features": list(ppe_model.FEATURES),
        "threshold": 0.5,
        "models": fit(train_rows, args.seed),
        "metrics": {},
        "training": {"rows": len(rows), "test_rows": len(test_rows), "seed": args.seed,
                     "sklearn": sklearn.__version__},
    }
    # Evaluate the exported (rounded) parameters through the same code the backend serves with.
    candidate = evaluate(ppe_model.LogisticModel(spec), test_rows)
    spec["metrics"] = candidate["metrics"]
    rule = evaluate(ppe_model.RuleModel(), test_rows)
    champion = load_previous(out_path)
    previous = evaluate(champion, test_rows) if champion else None

    for item_type in ppe_model.ITEM_TYPES + ("overall",):
        m, r = candidate["metrics"][item_type], rule["metrics"][item_type]
        log("{:8} accuracy {:.4f}  f1 {:.4f}   (overlap rule: accuracy {:.4f}  f1 {:.4f})".format(
            item_type, m["accuracy"], m["f1"], r["accuracy"], r["f1"]))
    if previous:
        log("Champion {} overall F1 {:.4f} vs challenger {} overall F1 {:.4f} (max regression {:.4f})".format(
            previous["version"], previous["metrics"]["overall"]["f1"], spec["version"],
            candidate["metrics"]["overall"]["f1"], args.max_regression))

    reasons = check_gate(candidate, previous, args.min_accuracy, args.max_regression)
    passed = not reasons
    if passed:
        _write_json(out_path, spec)
    report = {
        "generated_at": spec["trained_at"],
        "data": {"path": _show(data_path), "sha256": sha, "rows": len(rows), "train_rows": len(train_rows),
                 "test_rows": len(test_rows), "seed": args.seed},
        "candidate": candidate,
        "baseline_rule": rule,
        "previous_model": previous,
        "gate": {"passed": passed, "min_accuracy": args.min_accuracy,
                 "max_regression": args.max_regression, "reasons": reasons},
        "model_written": _show(out_path) if passed else None,
    }
    _write_json(report_path, report)
    log("Report written to {}".format(_show(report_path)))
    if args.summary:
        summary_path = os.path.join(ROOT, args.summary)
        if os.path.dirname(summary_path):
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(_summary(report))

    if not passed:
        for reason in reasons:
            log("Quality gate FAILED: " + reason)
        log("Model NOT written; {} is unchanged".format(_show(out_path)))
        return 1
    log("Quality gate PASSED (every item accuracy >= {:.2f}, F1 regression <= {:.2f})".format(
        args.min_accuracy, args.max_regression))
    log("Wrote model {} to {}".format(spec["version"], _show(out_path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
