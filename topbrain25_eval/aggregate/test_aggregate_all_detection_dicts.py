import pandas as pd
from aggregate_all_detection_dicts import (
    aggregate_all_detection_dicts,
    count_sideroad_detection,
    get_dect_avg,
)
from topbrain25_eval.constants import (
    MUL_CLASS_LABEL_MAP_CT,
    MUL_CLASS_LABEL_MAP_MR,
    SIDEROAD_COMPONENT_LABELS_CT,
    SIDEROAD_COMPONENT_LABELS_MR,
    TRACK,
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
# List of dictionaries
input_dicts_ct = [
    Fig50_small_multiclass_common
    | {
        # only in CT
        "37": {"label": "ICVs", "Detection": "TN"},
        "38": {"label": "R-BVR", "Detection": "TN"},
        "39": {"label": "L-BVR", "Detection": "TN"},
    },
    ThresholdIoU_common
    | {
        # only in CT
        "37": {"label": "ICVs", "Detection": "TN"},
        "38": {"label": "R-BVR", "Detection": "TN"},
        "39": {"label": "L-BVR", "Detection": "TN"},
    },
]
# Create a Pandas Series with dictionaries
all_detection_dicts_ct = pd.Series(input_dicts_ct)

input_dicts_mr = [
    Fig50_small_multiclass_common
    | {
        # only in MR
        "41": {"label": "R-MMA", "Detection": "TN"},
        "42": {"label": "L-MMA", "Detection": "TN"},
    },
    ThresholdIoU_common
    | {
        # only in MR
        "41": {"label": "R-MMA", "Detection": "TN"},
        "42": {"label": "L-MMA", "Detection": "TN"},
    },
]
# Create a Pandas Series with dictionaries
all_detection_dicts_mr = pd.Series(input_dicts_mr)

# counts from test_count_sideroad_detection_ct()
expected_detection_counts_ct = {
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
    # CT only > 34
    "37": {"label": "ICVs", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "38": {"label": "R-BVR", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "39": {"label": "L-BVR", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
}

# counts from test_count_sideroad_detection_mr()
expected_detection_counts_mr = {
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
    # MR only labels
    "41": {"label": "R-MMA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
    "42": {"label": "L-MMA", "TP": 0, "TN": 2, "FP": 0, "FN": 0},
}

expected_aggre_detect_ct_result = {
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
    "37": {"label": "ICVs", "precision": 0, "recall": 0, "f1_score": 0},
    "38": {"label": "R-BVR", "precision": 0, "recall": 0, "f1_score": 0},
    "39": {"label": "L-BVR", "precision": 0, "recall": 0, "f1_score": 0},
    "precision": {"mean": 0.1388888888888889, "std": 0.32513055307554517},
    "recall": {"mean": 0.16666666666666666, "std": 0.37267799624996495},
    "f1_score": {"mean": 0.14814814814814814, "std": 0.33742346589423333},
}


def test_count_sideroad_detection_ct():
    assert (
        count_sideroad_detection(
            all_detection_dicts_ct, SIDEROAD_COMPONENT_LABELS_CT, MUL_CLASS_LABEL_MAP_CT
        )
        == expected_detection_counts_ct
    )


def test_count_sideroad_detection_mr():
    assert (
        count_sideroad_detection(
            all_detection_dicts_mr, SIDEROAD_COMPONENT_LABELS_MR, MUL_CLASS_LABEL_MAP_MR
        )
        == expected_detection_counts_mr
    )


def test_get_dect_avg():
    assert (
        get_dect_avg(
            expected_detection_counts_ct,
            SIDEROAD_COMPONENT_LABELS_CT,
            MUL_CLASS_LABEL_MAP_CT,
        )
        == expected_aggre_detect_ct_result
    )

    ####################################################################################
    # counts from
    # https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
    google_ml_detection_counts_mr = {
        # A model outputs 5 TP, 6 TN, 3 FP, and 2 FN. Calculate the recall.
        "8": {"label": "R-Pcom", "TP": 5, "TN": 6, "FP": 3, "FN": 2},
        # A model outputs 3 TP, 4 TN, 2 FP, and 1 FN. Calculate the precision.
        "9": {"label": "L-Pcom", "TP": 3, "TN": 4, "FP": 2, "FN": 1},
        # Precision 0.85, Recall 0.83
        "10": {"label": "Acom", "TN": 44, "TP": 40, "FN": 8, "FP": 7},
        # Precision 0.97, Recall 0.63
        "15": {"label": "3rd-A2", "TN": 50, "TP": 30, "FN": 18, "FP": 1},
        # the rest all 2025 Precision -> 2/4; Recall -> 2/7
        "16": {"label": "3rd-A3", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        "25": {"label": "R-SCA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        "26": {"label": "L-SCA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        "27": {"label": "R-AICA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        "28": {"label": "L-AICA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        "29": {"label": "R-PICA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        "30": {"label": "L-PICA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        "31": {"label": "R-AChA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        "32": {"label": "L-AChA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        "33": {"label": "R-OA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        "34": {"label": "L-OA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        # MR only labels
        "41": {"label": "R-MMA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
        "42": {"label": "L-MMA", "TP": 2, "TN": 0, "FP": 2, "FN": 5},
    }
    assert get_dect_avg(
        google_ml_detection_counts_mr,
        SIDEROAD_COMPONENT_LABELS_MR,
        MUL_CLASS_LABEL_MAP_MR,
    ) == {
        # Recall is calculated as [\frac{TP}{TP+FN}=\frac{5}{7}]. = 0.714
        "8": {
            "label": "R-Pcom",
            "precision": 0.625,
            "recall": 0.7142857142857143,
            "f1_score": 0.6666666666666666,
        },
        # Precision is calculated as [\frac{TP}{TP+FP}=\frac{3}{5}]. = 0.6
        "9": {
            "label": "L-Pcom",
            "precision": 0.6,
            "recall": 0.75,
            "f1_score": 0.6666666666666666,
        },
        # Precision 0.85, Recall 0.83
        "10": {
            "label": "Acom",
            "precision": 0.851063829787234,
            "recall": 0.8333333333333334,
            "f1_score": 0.8421052631578947,
        },
        # Precision 0.97, Recall 0.63
        "15": {
            "label": "3rd-A2",
            "precision": 0.967741935483871,
            "recall": 0.625,
            "f1_score": 0.759493670886076,
        },
        "16": {
            "label": "3rd-A3",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "25": {
            "label": "R-SCA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "26": {
            "label": "L-SCA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "27": {
            "label": "R-AICA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "28": {
            "label": "L-AICA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "29": {
            "label": "R-PICA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "30": {
            "label": "L-PICA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "31": {
            "label": "R-AChA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "32": {
            "label": "L-AChA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "33": {
            "label": "R-OA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "34": {
            "label": "L-OA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "41": {
            "label": "R-MMA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "42": {
            "label": "L-MMA",
            "precision": 0.5,
            "recall": 0.2857142857142857,
            "f1_score": 0.36363636363636365,
        },
        "precision": {"mean": 0.5614003391335944, "std": 0.13362883248624569},
        "recall": {"mean": 0.39040616246498594, "std": 0.1921870255050573},
        "f1_score": {"mean": 0.4507179408617666, "std": 0.16094862279558578},
    }


def test_aggregate_all_detection_dicts():
    """
    combine test_count_sideroad_detection()
    and test_get_dect_avg()
    """
    assert (
        aggregate_all_detection_dicts(TRACK.CT, all_detection_dicts_ct)
        == expected_aggre_detect_ct_result
    )

    assert aggregate_all_detection_dicts(TRACK.MR, all_detection_dicts_mr) == {
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
        "41": {"label": "R-MMA", "precision": 0, "recall": 0, "f1_score": 0},
        "42": {"label": "L-MMA", "precision": 0, "recall": 0, "f1_score": 0},
        "precision": {"mean": 0.14705882352941177, "std": 0.3327561323230812},
        "recall": {"mean": 0.17647058823529413, "std": 0.38122004108281526},
        "f1_score": {"mean": 0.1568627450980392, "std": 0.34523170316978447},
    }
