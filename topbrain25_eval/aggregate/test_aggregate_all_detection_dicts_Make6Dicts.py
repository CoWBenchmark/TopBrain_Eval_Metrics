import pandas as pd
from aggregate_all_detection_dicts import (
    aggregate_all_detection_dicts,
    count_sideroad_detection,
    get_dect_avg,
)
from pytest import approx
from topbrain25_eval.constants import (
    MUL_CLASS_LABEL_MAP,
    SIDEROAD_COMPONENT_LABELS,
)


def make_dict(detection, overrides=None):
    labels = {
        "8": "R-Pcom",
        "9": "L-Pcom",
        "10": "Acom",
        "15": "3rd-A2",
        "16": "3rd-A3",
        "25": "R-SCA",
        "26": "L-SCA",
        "27": "R-AICA",
        "28": "L-AICA",
        "29": "R-PICA",
        "30": "L-PICA",
        "31": "R-AChA",
        "32": "L-AChA",
        "33": "R-OA",
        "34": "L-OA",
    }
    if overrides is None:
        overrides = {}
    return {
        key: {
            "label": label,
            "Detection": overrides.get(key, detection),
        }
        for key, label in labels.items()
    }


# Create a Pandas Series with List of dictionaries
all_detection_dicts = pd.Series(
    # 2xTP, 1xTN, 1xFP, 2xFN, with some overrides.
    [
        make_dict("TP", {"31": "TN"}),
        make_dict("TN", {"34": "TP"}),
        make_dict("FP"),
        make_dict("TP"),
        make_dict("FN"),
        make_dict("FN", {"8": "TP"}),
    ]
)
# print out for debug
for i, detection_dict in all_detection_dicts.items():
    print(f"\n--- dict {i} ---")
    for key, value in detection_dict.items():
        print(key, value)

common_count = {"TP": 2, "FP": 1, "TN": 1, "FN": 2}
expected_detection_counts = {
    "8": {"label": "R-Pcom", "TP": 3, "FP": 1, "TN": 1, "FN": 1},
    "9": {"label": "L-Pcom", **common_count},
    "10": {"label": "Acom", **common_count},
    "15": {"label": "3rd-A2", **common_count},
    "16": {"label": "3rd-A3", **common_count},
    "25": {"label": "R-SCA", **common_count},
    "26": {"label": "L-SCA", **common_count},
    "27": {"label": "R-AICA", **common_count},
    "28": {"label": "L-AICA", **common_count},
    "29": {"label": "R-PICA", **common_count},
    "30": {"label": "L-PICA", **common_count},
    "31": {"label": "R-AChA", "TP": 1, "FP": 1, "TN": 2, "FN": 2},
    "32": {"label": "L-AChA", **common_count},
    "33": {"label": "R-OA", **common_count},
    "34": {"label": "L-OA", "TP": 3, "FP": 1, "TN": 0, "FN": 2},
}

commom_aggre = {"precision": 2 / 3, "recall": 0.5, "f1_score": 4 / 7}
expected_aggre_detect_result = {
    "8": {"label": "R-Pcom", "precision": 0.75, "recall": 0.75, "f1_score": 0.75},
    "9": {"label": "L-Pcom", **commom_aggre},
    "10": {"label": "Acom", **commom_aggre},
    "15": {"label": "3rd-A2", **commom_aggre},
    "16": {"label": "3rd-A3", **commom_aggre},
    "25": {"label": "R-SCA", **commom_aggre},
    "26": {"label": "L-SCA", **commom_aggre},
    "27": {"label": "R-AICA", **commom_aggre},
    "28": {"label": "L-AICA", **commom_aggre},
    "29": {"label": "R-PICA", **commom_aggre},
    "30": {"label": "L-PICA", **commom_aggre},
    "31": {"label": "R-AChA", "precision": 0.5, "recall": 1 / 3, "f1_score": 0.4},
    "32": {"label": "L-AChA", **commom_aggre},
    "33": {"label": "R-OA", **commom_aggre},
    "34": {"label": "L-OA", "precision": 0.75, "recall": 0.6, "f1_score": 2 / 3},
    # mean and std
    "precision": {"mean": 2 / 3, "std": 0.05270462766947299},
    "recall": {"mean": 0.5122222222222221, "std": 0.08084431006036111},
    "f1_score": {"mean": 0.5782539682539681, "std": 0.0681405480661356},
}


def _assert_nested_dict_approx(actual, expected):
    """Create a small helper that applies approx() only to the actual numeric values"""
    assert actual.keys() == expected.keys()

    for key, expected_value in expected.items():
        actual_value = actual[key]

        if isinstance(expected_value, dict):
            _assert_nested_dict_approx(actual_value, expected_value)
        elif isinstance(expected_value, float):
            assert actual_value == approx(expected_value)
        else:
            assert actual_value == expected_value


def test_count_sideroad_detection_Make6Dicts():
    actual = count_sideroad_detection(
        all_detection_dicts, SIDEROAD_COMPONENT_LABELS, MUL_CLASS_LABEL_MAP
    )
    _assert_nested_dict_approx(
        actual,
        expected_detection_counts,
    )


def test_get_dect_avg_Make6Dicts():
    actual = get_dect_avg(
        expected_detection_counts,
        SIDEROAD_COMPONENT_LABELS,
        MUL_CLASS_LABEL_MAP,
    )
    _assert_nested_dict_approx(
        actual,
        expected_aggre_detect_result,
    )


def test_aggregate_all_detection_dicts_Make6Dicts():
    """
    combine test_count_sideroad_detection_Make6Dicts()
    and test_get_dect_avg_Make6Dicts()
    """
    actual = aggregate_all_detection_dicts(all_detection_dicts)
    _assert_nested_dict_approx(
        actual,
        expected_aggre_detect_result,
    )
