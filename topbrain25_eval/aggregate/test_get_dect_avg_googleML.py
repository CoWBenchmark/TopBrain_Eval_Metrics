from aggregate_all_detection_dicts import (
    get_dect_avg,
)
from topbrain25_eval.constants import (
    MUL_CLASS_LABEL_MAP,
    SIDEROAD_COMPONENT_LABELS,
)


def test_get_dect_avg_googleML():
    ####################################################################################
    # counts from
    # https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
    google_ml_detection_counts = {
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
    }
    assert get_dect_avg(
        google_ml_detection_counts,
        SIDEROAD_COMPONENT_LABELS,
        MUL_CLASS_LABEL_MAP,
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
        # "41": {
        #     "label": "R-MMA",
        #     "precision": 0.5,
        #     "recall": 0.2857142857142857,
        #     "f1_score": 0.36363636363636365,
        # },
        # "42": {
        #     "label": "L-MMA",
        #     "precision": 0.5,
        #     "recall": 0.2857142857142857,
        #     "f1_score": 0.36363636363636365,
        # },
        "precision": {
            "mean": (0.5614003391335944 * 17 - 1) / 15,
            "std": 0.1402421633590027,
        },
        "recall": {"mean": 0.4043650793650793, "std": 0.2005103403155164},
        "f1_score": {"mean": 0.4623288178251535, "std": 0.16796571414319778},
    }
