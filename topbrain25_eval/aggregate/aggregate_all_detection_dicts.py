"""
aggregate the detection_dict
from pandas DataFrame: self._case_results
Get the Series: self._case_results["all_detection_dicts"]
detection_dict is under the column `all_detection_dicts`

to get the Average F1 score
(harmonic mean of the precision and recall)
for detection of the "Side road" vessel components
"""

import pprint

import numpy as np
from pandas import Series
from topbrain25_eval.constants import (
    MUL_CLASS_LABEL_MAP_CT,
    MUL_CLASS_LABEL_MAP_MR,
    SIDEROAD_COMPONENT_LABELS_CT,
    SIDEROAD_COMPONENT_LABELS_MR,
    TRACK,
)


def aggregate_all_detection_dicts(track: TRACK, all_detection_dicts: Series) -> dict:
    print("\n[aggregate] aggregate_all_detection_dicts()\n")

    if track == TRACK.CT:
        component_labels = SIDEROAD_COMPONENT_LABELS_CT
        label_map = MUL_CLASS_LABEL_MAP_CT
    else:
        component_labels = SIDEROAD_COMPONENT_LABELS_MR
        label_map = MUL_CLASS_LABEL_MAP_MR

    # get the detection counts
    detection_counts = count_sideroad_detection(
        all_detection_dicts, component_labels, label_map
    )
    # get the detection averages
    dect_avg = get_dect_avg(detection_counts, component_labels, label_map)

    return dect_avg


def count_sideroad_detection(
    all_detection_dicts: Series, component_labels: tuple, label_map: dict
) -> dict:
    """
    input all_detection_dicts is a pandas.Series of detection_dicts
    return the count of TP|TN|FP|FN from all detection_dicts

    init the side-road detection stats dictionary
    with label name and four 0 detection scores,
    then count the number for each detection category
    """
    # first sanity check: the number of labels in component_labels must
    # be the same as the number of items in a detection_dict
    num_component_labels = len(component_labels)
    num_detection_dict_labels = len(all_detection_dicts.values[0])
    print(f"num_component_labels = {num_component_labels}")
    print(f"num_detection_dict_labels = {num_detection_dict_labels}")

    assert num_component_labels == num_detection_dict_labels, (
        "unmatched num of labels for detection metric"
    )

    detection_counts = {}

    for label in component_labels:
        detection_counts[str(label)] = {
            "label": label_map[str(label)],
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
        }

    # get the values from the pandas Series
    for detection_dict in all_detection_dicts.values:
        # each detection_dict is a dictionary
        # type(detection_dict):  <class 'dict'>

        for label, value in detection_dict.items():
            detection_counts[label][value["Detection"]] += 1

    print("\ncount_sideroad_detection =>")
    pprint.pprint(detection_counts, sort_dicts=False)

    return detection_counts


def get_dect_avg(
    detection_counts: dict, component_labels: tuple, label_map: dict
) -> dict:
    """
    calculate the mean and std of detections for side-road vessels
    compute for precision, recall, and f1_score
    also record down the detections for each side-road class
    treat NaN as 0 during averaging for recall, precision, and f1

    Returns:
        dect_avg: dictionary

        {
            "8": {
                "label": "R-Pcom",
                "precision": 0.625,
                "recall": 0.7142857142857143,
                "f1_score": 0.6666666666666666,
            },
            "9": {...},
            "10": {...},
            "15": {...},
            "precision": {
                "mean": ...,
                "std": ...
            },
            "recall": {"mean": ..., "std": ...},
            "f1_score": {"mean": ..., "std": ...},
        }
    """
    dect_avg = {}

    # init the dect_avg dict with entries
    for label in component_labels:
        dect_avg[str(label)] = {
            "label": label_map[str(label)],
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
        }

    list_recall = []
    list_precision = []
    list_f1 = []

    # get the detection_stats based on label
    for label, stats in detection_counts.items():
        print(f"\nfor label-{label} ({label_map[str(label)]})")
        tp = stats["TP"]
        fp = stats["FP"]
        fn = stats["FN"]

        # handle the undefined division by zero
        # NOTE: treat the nan as 0

        # Precision
        if (tp + fp) == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        print(f"precision = {precision}")
        list_precision.append(precision)
        dect_avg[label]["precision"] = precision

        # Recall
        if (tp + fn) == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        print(f"recall = {recall}")
        list_recall.append(recall)
        dect_avg[label]["recall"] = recall

        # F1
        if (tp + fp + fn) == 0:
            f1_score = 0
        else:
            f1_score = 2 * tp / ((2 * tp) + fp + fn)

        print(f"f1_score = {f1_score}")
        list_f1.append(f1_score)
        dect_avg[label]["f1_score"] = f1_score

    # end of for loop

    print(f"\nlist_precision = {list_precision}")
    dect_avg["precision"] = {
        "mean": np.mean(list_precision),
        "std": np.std(list_precision),
    }

    print(f"\nlist_recall = {list_recall}")
    dect_avg["recall"] = {
        "mean": np.mean(list_recall),
        "std": np.std(list_recall),
    }

    print(f"\nlist_f1 = {list_f1}")
    dect_avg["f1_score"] = {
        "mean": np.mean(list_f1),
        "std": np.std(list_f1),
    }

    print("\nget_dect_avg =>")
    pprint.pprint(dect_avg, sort_dicts=False)

    return dect_avg
