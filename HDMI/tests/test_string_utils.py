import pytest

from active_adaptation.utils.string import (
    resolve_matching_names,
    resolve_matching_names_values,
)


NAMES = ["left_hip", "torso", "left_knee", "right_hip"]


def test_resolve_matching_names_returns_target_order_by_default():
    indices, names = resolve_matching_names(["left_.*", "torso"], NAMES)

    assert indices == [0, 1, 2]
    assert names == ["left_hip", "torso", "left_knee"]


def test_resolve_matching_names_can_preserve_query_order():
    indices, names = resolve_matching_names(["left_.*", "torso"], NAMES, preserve_order=True)

    assert indices == [0, 2, 1]
    assert names == ["left_hip", "left_knee", "torso"]


def test_resolve_matching_names_rejects_unmatched_regex():
    with pytest.raises(ValueError, match="Not all regular expressions are matched"):
        resolve_matching_names(["left_.*", "missing"], NAMES)


def test_resolve_matching_names_rejects_ambiguous_regexes():
    with pytest.raises(ValueError, match="Multiple matches for 'left_hip'"):
        resolve_matching_names(["left_.*", ".*hip"], NAMES)


def test_resolve_matching_names_values_returns_target_order_by_default():
    indices, names, values = resolve_matching_names_values(
        {"left_.*": 0.5, "torso": 1.0},
        NAMES,
    )

    assert indices == [0, 1, 2]
    assert names == ["left_hip", "torso", "left_knee"]
    assert values == [0.5, 1.0, 0.5]


def test_resolve_matching_names_values_can_preserve_query_order():
    indices, names, values = resolve_matching_names_values(
        {"left_.*": 0.5, "torso": 1.0},
        NAMES,
        preserve_order=True,
    )

    assert indices == [0, 2, 1]
    assert names == ["left_hip", "left_knee", "torso"]
