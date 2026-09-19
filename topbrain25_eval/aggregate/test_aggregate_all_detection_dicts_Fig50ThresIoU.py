import pandas as pd
from aggregate_all_detection_dicts import (
    aggregate_all_detection_dicts,
    count_sideroad_detection,
    get_dect_avg,
)
from topbrain25_eval.constants import (
    MUL_CLASS_LABEL_MAP,
    SIDEROAD_COMPONENT_LABELS,
)

# reuse detection_dict from metrics/test_detection_sideroad_labels.py
# from test_detection_sideroad_labels_Fig50_small_multiclass()
# from test_detection_sideroad_labels_ThresholdIoU()

Fig50_small_multiclass_common = {
    "8": {"label": "R-Pcom", "Detection": "TN"},
    "9": {"label": "L-Pcom", "Detection": "TN"},
    "10": {"label": "Acom", "Detection": "TP"},
    "15": {"label": "3rd-A2", "Detection": "TP"},
    # new in topbrain, all absent in both GT and Pred, thus TN
    "16": {"label": "3rd-A3", "Detection": "TN"},
    "25": {"label": "R-SCA", "Detection": "TN"},
    "26": {"label": "L-SCA", "Detection": "TN"},
    "27": {"label": "R-AICA", "Detection": "TN"},
    "28": {"label": "L-AICA", "Detection": "TN"},
    "29": {"label": "R-PICA", "Detection": "TN"},
    "30": {"label": "L-PICA", "Detection": "TN"},
    "31": {"label": "R-AChA", "Detection": "TN"},
    "32": {"label": "L-AChA", "Detection": "TN"},
    "33": {"label": "R-OA", "Detection": "TN"},
    "34": {"label": "L-OA", "Detection": "TN"},
}
ThresholdIoU_common = {
    # label-8 IoU = 0.25 -> TP
    "8": {"label": "R-Pcom", "Detection": "TP"},
    # label-9 IoU < 0.25 -> FN
    "9": {"label": "L-Pcom", "Detection": "FN"},
    # label-10 IoU > 0.25 -> TP
    "10": {"label": "Acom", "Detection": "TP"},
    # label-15 GT missing, pred not -> FP
    "15": {"label": "3rd-A2", "Detection": "FP"},
    # remaining side road vessels, all TN
    "16": {"label": "3rd-A3", "Detection": "TN"},
    "25": {"label": "R-SCA", "Detection": "TN"},
    "26": {"label": "L-SCA", "Detection": "TN"},
    "27": {"label": "R-AICA", "Detection": "TN"},
    "28": {"label": "L-AICA", "Detection": "TN"},
    "29": {"label": "R-PICA", "Detection": "TN"},
    "30": {"label": "L-PICA", "Detection": "TN"},
    "31": {"label": "R-AChA", "Detection": "TN"},
    "32": {"label": "L-AChA", "Detection": "TN"},
    "33": {"label": "R-OA", "Detection": "TN"},
    "34": {"label": "L-OA", "Detection": "TN"},
}

# Create a Pandas Series with List of dictionaries
all_detection_dicts = pd.Series(
    [
        Fig50_small_multiclass_common,
        ThresholdIoU_common,
    ]
)

# counts from test_count_sideroad_detection_Fig50ThresIoU()
expected_detection_counts = {
    "8": {"label": "R-Pcom", "TN": 1, "TP": 1, "FN": 0, "FP": 0},
    "9": {"label": "L-Pcom", "TN": 1, "TP": 0, "FN": 1, "FP": 0},
    "10": {"label": "Acom", "TN": 0, "TP": 2, "FN": 0, "FP": 0},
    "15": {"label": "3rd-A2", "TN": 0, "TP": 1, "FN": 0, "FP": 1},
    # the rest all TN = 2 due to two dicts
    "16": {"label": "3rd-A3", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "25": {"label": "R-SCA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "26": {"label": "L-SCA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "27": {"label": "R-AICA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "28": {"label": "L-AICA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "29": {"label": "R-PICA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "30": {"label": "L-PICA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "31": {"label": "R-AChA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "32": {"label": "L-AChA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "33": {"label": "R-OA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "34": {"label": "L-OA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
}

expected_aggre_detect_result = {
    "8": {"label": "R-Pcom", "precision": 1.0, "recall": 1.0, "f1_score": 1.0},
    "9": {"label": "L-Pcom", "precision": 0, "recall": 0.0, "f1_score": 0.0},
    "10": {"label": "Acom", "precision": 1.0, "recall": 1.0, "f1_score": 1.0},
    "15": {
        "label": "3rd-A2",
        "precision": 0.5,
        "recall": 1.0,
        "f1_score": 0.6666666666666666,
    },
    "16": {"label": "3rd-A3", "precision": 0, "recall": 0, "f1_score": 0},
    "25": {"label": "R-SCA", "precision": 0, "recall": 0, "f1_score": 0},
    "26": {"label": "L-SCA", "precision": 0, "recall": 0, "f1_score": 0},
    "27": {"label": "R-AICA", "precision": 0, "recall": 0, "f1_score": 0},
    "28": {"label": "L-AICA", "precision": 0, "recall": 0, "f1_score": 0},
    "29": {"label": "R-PICA", "precision": 0, "recall": 0, "f1_score": 0},
    "30": {"label": "L-PICA", "precision": 0, "recall": 0, "f1_score": 0},
    "31": {"label": "R-AChA", "precision": 0, "recall": 0, "f1_score": 0},
    "32": {"label": "L-AChA", "precision": 0, "recall": 0, "f1_score": 0},
    "33": {"label": "R-OA", "precision": 0, "recall": 0, "f1_score": 0},
    "34": {"label": "L-OA", "precision": 0, "recall": 0, "f1_score": 0},
    # "37": {"label": "ICVs", "precision": 0, "recall": 0, "f1_score": 0},
    # "38": {"label": "R-BVR", "precision": 0, "recall": 0, "f1_score": 0},
    # "39": {"label": "L-BVR", "precision": 0, "recall": 0, "f1_score": 0},
    "precision": {"mean": 0.1388888888888889 * 18 / 15, "std": 0.3496029493900505},
    "recall": {"mean": 0.16666666666666666 * 18 / 15, "std": 0.4000000000000001},
    "f1_score": {"mean": 0.14814814814814814 * 18 / 15, "std": 0.36243347622889094},
}


def test_count_sideroad_detection_Fig50ThresIoU():
    assert (
        count_sideroad_detection(
            all_detection_dicts, SIDEROAD_COMPONENT_LABELS, MUL_CLASS_LABEL_MAP
        )
        == expected_detection_counts
    )


def test_get_dect_avg_Fig50ThresIoU():
    assert (
        get_dect_avg(
            expected_detection_counts,
            SIDEROAD_COMPONENT_LABELS,
            MUL_CLASS_LABEL_MAP,
        )
        == expected_aggre_detect_result
    )


def test_aggregate_all_detection_dicts_Fig50ThresIoU():
    """
    combine test_count_sideroad_detection_Fig50ThresIoU()
    and test_get_dect_avg_Fig50ThresIoU()
    """
    assert (
        aggregate_all_detection_dicts(all_detection_dicts)
        == expected_aggre_detect_result
    )
