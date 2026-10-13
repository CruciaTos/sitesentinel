"""Tests for the PPE association model as the backend serves it (pure Python, no numpy)."""
import json
import re

import pytest

import ppe_model

PERSON = [0.2, 0.4, 0.8, 0.6]  # height 0.6, width 0.2
HAT_ON_HEAD = [0.22, 0.47, 0.28, 0.53]
HAT_AT_WAIST = [0.5, 0.53, 0.56, 0.59]  # held in one hand, still inside the person's box
HAT_ON_GROUND = [0.76, 0.52, 0.82, 0.62]
VEST_ON_TORSO = [0.35, 0.42, 0.53, 0.58]
VEST_BESIDE = [0.35, 0.65, 0.53, 0.81]


@pytest.fixture(scope="module")
def model():
    return ppe_model.load()


def write_spec(tmp_path, spec):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return str(path)


def valid_spec():
    params = {"mean": [0.0] * 7, "scale": [1.0] * 7, "coef": [0.0] * 7, "intercept": 0.0}
    return {"version": "2026.01.01-abcdef0", "features": list(ppe_model.FEATURES),
            "models": {"hardhat": dict(params), "vest": dict(params)}}


@pytest.mark.parametrize("person, item, item_type, expected", [
    # Hat fully inside a 1.0 x 0.5 person, centred horizontally near the top.
    ([0.0, 0.0, 1.0, 0.5], [0.0, 0.2, 0.1, 0.3], "hardhat", [1.0, 0.02, 0.05, 0.03, 0.0, 0.2, 0.1]),
    # Vest overlapping the bottom-right corner: a quarter of it inside.
    ([0.2, 0.2, 0.6, 0.4], [0.5, 0.3, 0.7, 0.5], "vest", [0.25, 0.01 / 0.11, 1.0, 0.6, 0.5, 1.0, 0.5]),
    # Hat nowhere near the person.
    ([0.0, 0.0, 0.5, 0.5], [0.6, 0.6, 0.7, 0.7], "hardhat", [0.0, 0.0, 1.3, 1.22, 0.8, 0.2, 0.2]),
    # Far away from a small person: positions are clipped to POSITION_RANGE.
    ([0.0, 0.0, 0.1, 0.1], [0.8, 0.8, 0.9, 0.9], "hardhat", [0.0, 0.0, 2.0, 1.92, 1.5, 1.0, 1.0]),
    # A vest much bigger than the person: sizes are capped at MAX_REL_SIZE.
    ([0.0, 0.0, 0.1, 0.05], [0.0, 0.0, 0.5, 0.5], "vest", [0.02, 0.02, 2.0, 1.6, 1.5, 3.0, 3.0]),
])
def test_features_match_hand_computed_values(person, item, item_type, expected):
    assert ppe_model.features(person, item, item_type) == pytest.approx(expected)


def test_features_order_and_edge_cases():
    assert len(ppe_model.features(PERSON, HAT_ON_HEAD, "hardhat")) == len(ppe_model.FEATURES) == 7
    flat = ppe_model.features(PERSON, [0.3, 0.45, 0.3, 0.5], "vest")
    assert flat[0] == 0.0 and flat[1] == 0.0
    with pytest.raises(ValueError):
        ppe_model.features(PERSON, HAT_ON_HEAD, "gloves")


def test_rule_model_uses_the_overlap_rule():
    rule = ppe_model.RuleModel()
    assert rule.version == "rules-0.6" and rule.algorithm == "overlap_rule"
    assert rule.wears(PERSON, HAT_ON_HEAD, "hardhat") is True
    assert rule.wears(PERSON, VEST_BESIDE, "vest") is False
    # The weakness the trained model fixes: a hat held at the waist is inside the person's box.
    assert rule.wears(PERSON, HAT_AT_WAIST, "hardhat") is True


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '"text"', "{}"])
def test_load_falls_back_to_rule_for_corrupt_files(tmp_path, content):
    path = tmp_path / "model.json"
    path.write_text(content, encoding="utf-8")
    assert isinstance(ppe_model.load(str(path)), ppe_model.RuleModel)


def test_load_falls_back_to_rule_for_missing_file(tmp_path):
    model = ppe_model.load(str(tmp_path / "missing.json"))
    assert isinstance(model, ppe_model.RuleModel)
    assert model.info()["version"] == "rules-0.6"


def test_wrong_feature_list_is_rejected(tmp_path):
    spec = valid_spec()
    assert isinstance(ppe_model.load(write_spec(tmp_path, spec)), ppe_model.LogisticModel)
    spec["features"] = list(reversed(ppe_model.FEATURES))
    with pytest.raises(ValueError):
        ppe_model.LogisticModel(spec)
    assert isinstance(ppe_model.load(write_spec(tmp_path, spec)), ppe_model.RuleModel)


def test_wrong_number_of_weights_is_rejected():
    spec = valid_spec()
    spec["models"]["vest"]["coef"] = [0.0] * 6
    with pytest.raises(ValueError):
        ppe_model.LogisticModel(spec)


def test_committed_model_loads(model):
    assert isinstance(model, ppe_model.LogisticModel)
    assert re.match(r"^\d{4}\.\d{2}\.\d{2}-[0-9a-f]{7}$", model.version)
    assert model.algorithm == "logistic_regression"
    assert model.threshold == 0.5
    info = model.info()
    assert set(info) == {"name", "version", "algorithm", "trained_at", "metrics"}
    assert info["name"] == "ppe-association"
    assert info["metrics"]["overall"]["accuracy"] >= 0.9


@pytest.mark.parametrize("item, item_type, worn", [
    (HAT_ON_HEAD, "hardhat", True),
    (HAT_ON_GROUND, "hardhat", False),
    (HAT_AT_WAIST, "hardhat", False),
    (VEST_ON_TORSO, "vest", True),
    (VEST_BESIDE, "vest", False),
])
def test_committed_model_gets_obvious_cases_right(model, item, item_type, worn):
    assert model.wears(PERSON, item, item_type) is worn
    probability = model.probability(PERSON, item, item_type)
    assert 0.0 <= probability <= 1.0


def test_assess_people_returns_one_entry_per_person(model):
    other = [0.1, 0.0, 0.5, 0.12]
    result = ppe_model.assess_people(model, [PERSON, other], [HAT_ON_HEAD], [VEST_BESIDE])
    assert result == [{"hardhat": True, "vest": False}, {"hardhat": False, "vest": False}]
    assert all(type(v) is bool for entry in result for v in entry.values())
    assert ppe_model.assess_people(model, [PERSON], [], []) == [{"hardhat": False, "vest": False}]
    assert ppe_model.assess_people(model, [], [HAT_ON_HEAD], [VEST_ON_TORSO]) == []
