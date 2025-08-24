"""
End-to-end test for the entire evaluation pipeline
more tests
"""

import json
from pathlib import Path

from topbrain25_eval.constants import TRACK
from topbrain25_eval.evaluation import TopBrainEvaluation

TESTDIR = Path("test_assets/")


def test_e2e_TopBrainEvaluation_Task_1_Seg_CT_2():
    """
    in:
    - ./test_assets/task_1_seg_predictions_ct_2
    - ./test_assets/task_1_seg_ground-truth_ct_2

    re-use cases from aggregate/test_aggregate_all_detection_dicts.py
        test_aggregate_all_detection_dicts

    reuse detection_dict from metrics/test_detection_sideroad_labels.py

    from test_detection_sideroad_labels_Fig50_small_multiclass()
        gt_path = TESTDIR_3D / "shape_2x2x2_3D_Fig50_label15_10_gt.nii.gz"
        pred_path = TESTDIR_3D / "shape_2x2x2_3D_Fig50_label15_10_pred.nii.gz"

    from test_detection_sideroad_labels_ThresholdIoU()
        gt_path = TESTDIR_3D / "detection_label8910_gt_squareL4.nii.gz"
        pred_path = TESTDIR_3D / "detection_label8910_pred_squareL4.nii.gz"
    """

    track = TRACK.CT

    expected_num_cases = 2

    # folder prefix to differentiate the tasks
    prefix = "task_1_seg_"

    # output_path for clean up
    output_path = Path(f"{prefix}output_test_e2e_TopBrainEvaluation_CT_2/")

    evalRun = TopBrainEvaluation(
        track,
        expected_num_cases,
        predictions_path=TESTDIR / f"{prefix}predictions_ct_2/",
        ground_truth_path=TESTDIR / f"{prefix}ground-truth_ct_2/",
        output_path=output_path,
    )

    # run the evaluation
    evalRun.evaluate()

    # read the saved metrics.json
    with open(output_path / "metrics.json") as f:
        generated_metrics_json = json.load(f)

    # some sanity checks
    # file is sorted by filename, so 0th file is detection_label8910_gt_squareL4

    ################# 0th-file #################
    # Dice scores
    assert generated_metrics_json["case"]["Dice_R-Pcom"]["0"] == 2 / 5  # 0.4
    assert generated_metrics_json["case"]["Dice_L-Pcom"]["0"] == 6 / 19  # ~0.316
    assert generated_metrics_json["case"]["Dice_Acom"]["0"] == 10 / 21  # ~0.476
    assert generated_metrics_json["case"]["Dice_3rd-A2"]["0"] == 0
    assert (
        round(generated_metrics_json["case"]["Dice_ClsAvgDice"]["0"], 3) == 0.298
    )  # ~0.298
    # B0 -> 1 error for label-15
    assert generated_metrics_json["case"]["B0err_3rd-A2"]["0"] == 1
    assert generated_metrics_json["case"]["B0err_ClsAvgB0err"]["0"] == 1 / 4
    # NbErr same as in test_invalid_neighbors_all_classes_detection_label8910()
    assert generated_metrics_json["case"]["NbErr_ClsAvgNbErr"]["0"] == 1

    ################# 1th-file #################
    # Dice scores
    assert generated_metrics_json["case"]["Dice_Acom"]["1"] == 4 / 5  # 0.8
    assert generated_metrics_json["case"]["Dice_3rd-A2"]["1"] == 2 / 4  # 0.5
    assert generated_metrics_json["case"]["Dice_ClsAvgDice"]["1"] == 0.65
    # B0 -> no B0 error
    assert generated_metrics_json["case"]["B0err_ClsAvgB0err"]["1"] == 0
    # NbErr -> no neighbor error
    assert generated_metrics_json["case"]["NbErr_ClsAvgNbErr"]["1"] == 0

    # aggregates
    # detection same as in test_aggregate_all_detection_dicts.py
    assert generated_metrics_json["aggregates"]["dect_avg"]["precision"] == {
        "mean": 0.1388888888888889,
        "std": 0.32513055307554517,
    }
    assert generated_metrics_json["aggregates"]["dect_avg"]["recall"] == {
        "mean": 0.16666666666666666,
        "std": 0.37267799624996495,
    }
    assert generated_metrics_json["aggregates"]["dect_avg"]["f1_score"] == {
        "mean": 0.14814814814814814,
        "std": 0.33742346589423333,
    }

    # average of above sanity checks
    assert generated_metrics_json["aggregates"]["Dice_R-Pcom"]["mean"] == 2 / 5
    assert generated_metrics_json["aggregates"]["Dice_L-Pcom"]["mean"] == 6 / 19
    assert round(generated_metrics_json["aggregates"]["Dice_Acom"]["mean"], 2) == round(
        (0.476 + 0.8) / 2, 2
    )  # ~0.638
    assert generated_metrics_json["aggregates"]["Dice_3rd-A2"]["mean"] == 1 / 4
    assert round(
        generated_metrics_json["aggregates"]["Dice_ClsAvgDice"]["mean"], 2
    ) == round((0.298 + 0.65) / 2, 2)  # ~0.474
    assert generated_metrics_json["aggregates"]["B0err_ClsAvgB0err"]["mean"] == 0.125
    assert generated_metrics_json["aggregates"]["NbErr_ClsAvgNbErr"]["mean"] == 0.5

    # compare the saved and expected metrics.json

    with open(
        TESTDIR / f"{prefix}output_ct_2/expected_e2e_test_ct_2_metrics.json"
    ) as f:
        expected_metrics_json = json.load(f)

    assert expected_metrics_json == generated_metrics_json

    print(f"expected_metrics_json =\n{json.dumps(expected_metrics_json, indent=2)}")
    print(f"generated_metrics_json =\n{json.dumps(generated_metrics_json, indent=2)}")

    # clean up the new metrics.json
    (output_path / "metrics.json").unlink()
    # clean up the output_path folder
    output_path.rmdir()


def test_e2e_TopBrainEvaluation_Task_1_Seg_MR_2():
    """
    in:
    - ./test_assets/task_1_seg_predictions_mr_2
    - ./test_assets/task_1_seg_ground-truth_mr_2

    re-use cases from test_detection_sideroad_labels_ThresholdIoU_MR
        gt_path = TESTDIR_3D / "detection_label8910_gt_squareL4_MR.nii.gz"
        pred_path = TESTDIR_3D / "detection_label8910_pred_squareL4_MR.nii.gz"
    & from test_clDice_all_classes
        gt_img = sitk.ReadImage(TESTDIR_3D / "shape_5x6x4_multiclass_clDice_gt.nii.gz")
        pred_img = sitk.ReadImage(TESTDIR_3D / "shape_5x6x4_multiclass_clDice_pred1.nii.gz")
    """

    track = TRACK.MR

    expected_num_cases = 2

    # folder prefix to differentiate the tasks
    prefix = "task_1_seg_"

    # output_path for clean up
    output_path = Path(f"{prefix}output_test_e2e_TopBrainEvaluation_MR_2/")

    evalRun = TopBrainEvaluation(
        track,
        expected_num_cases,
        predictions_path=TESTDIR / f"{prefix}predictions_mr_2/",
        ground_truth_path=TESTDIR / f"{prefix}ground-truth_mr_2/",
        output_path=output_path,
    )

    # run the evaluation
    evalRun.evaluate()

    # read the saved metrics.json
    with open(output_path / "metrics.json") as f:
        generated_metrics_json = json.load(f)

    # some sanity checks
    # file is sorted by filename, so 0th file is detection_label8910_gt_squareL4_MR

    ################# 0th-file #################
    # Dice scores same as 0th-case for test_e2e_TopBrainEvaluation_Task_1_Seg_CT_2
    # but the labels are now: 33,34,41,42
    assert generated_metrics_json["case"]["Dice_L-MMA"]["0"] == 2 / 5  # 0.4
    assert generated_metrics_json["case"]["Dice_R-MMA"]["0"] == 6 / 19  # ~0.316
    assert generated_metrics_json["case"]["Dice_L-OA"]["0"] == 10 / 21  # ~0.476
    assert generated_metrics_json["case"]["Dice_R-OA"]["0"] == 0
    assert (
        round(generated_metrics_json["case"]["Dice_ClsAvgDice"]["0"], 3) == 0.298
    )  # ~0.298
    # B0 -> 1 error for label-33
    assert generated_metrics_json["case"]["B0err_R-OA"]["0"] == 1
    assert generated_metrics_json["case"]["B0err_ClsAvgB0err"]["0"] == 1 / 4
    # NbErr same as in test_invalid_neighbors_all_classes_detection_label33344142()
    assert generated_metrics_json["case"]["NbErr_ClsAvgNbErr"]["0"] == 1.5
    # detection same as test_detection_sideroad_labels_ThresholdIoU_MR()
    assert (
        generated_metrics_json["case"]["all_detection_dicts"]["0"]["42"]["Detection"]
        == "TP"
    )
    assert (
        generated_metrics_json["case"]["all_detection_dicts"]["0"]["41"]["Detection"]
        == "FN"
    )
    assert (
        generated_metrics_json["case"]["all_detection_dicts"]["0"]["34"]["Detection"]
        == "TP"
    )
    assert (
        generated_metrics_json["case"]["all_detection_dicts"]["0"]["33"]["Detection"]
        == "FP"
    )

    ################# 1th-file #################
    # clDice same as in test_clDice_all_classes()
    assert generated_metrics_json["case"]["clDice_L-MaxA"]["1"] == 1
    assert round(generated_metrics_json["case"]["clDice_R-MaxA"]["1"], 3) == 0.857
    assert round(generated_metrics_json["case"]["clDice_L-STA"]["1"], 3) == 0.857
    assert round(generated_metrics_json["case"]["clDice_ClsAvgclDice"]["1"], 3) == 0.905
    # Dice scores
    assert generated_metrics_json["case"]["Dice_L-MaxA"]["1"] == 12 / 20  # 0.6
    assert generated_metrics_json["case"]["Dice_R-MaxA"]["1"] == 10 / 19  # ~0.526
    assert generated_metrics_json["case"]["Dice_L-STA"]["1"] == 14 / 15  # ~0.933
    assert (
        round(generated_metrics_json["case"]["Dice_ClsAvgDice"]["1"], 3) == 0.687
    )  # ~0.6865
    # B0 -> no B0 error
    assert generated_metrics_json["case"]["B0err_ClsAvgB0err"]["1"] == 0
    # NbErr -> all wrong neighbors
    assert generated_metrics_json["case"]["NbErr_ClsAvgNbErr"]["1"] == 4 / 3  # ~1.33

    # average of above sanity checks
    assert generated_metrics_json["aggregates"]["Dice_R-OA"]["mean"] == 0
    assert generated_metrics_json["aggregates"]["Dice_L-OA"]["mean"] == 10 / 21
    assert generated_metrics_json["aggregates"]["Dice_R-MMA"]["mean"] == 6 / 19
    assert generated_metrics_json["aggregates"]["Dice_L-MMA"]["mean"] == 2 / 5
    assert round(
        generated_metrics_json["aggregates"]["Dice_ClsAvgDice"]["mean"], 2
    ) == round((0.298 + 0.687) / 2, 2)  # ~0.4925
    assert generated_metrics_json["aggregates"]["B0err_ClsAvgB0err"]["mean"] == 0.125
    assert round(
        generated_metrics_json["aggregates"]["NbErr_ClsAvgNbErr"]["mean"], 2
    ) == round((1.5 + 4 / 3) / 2, 2)  # ~1.42

    # compare the saved and expected metrics.json

    with open(
        TESTDIR / f"{prefix}output_mr_2/expected_e2e_test_mr_2_metrics.json"
    ) as f:
        expected_metrics_json = json.load(f)

    assert expected_metrics_json == generated_metrics_json

    print(f"expected_metrics_json =\n{json.dumps(expected_metrics_json, indent=2)}")
    print(f"generated_metrics_json =\n{json.dumps(generated_metrics_json, indent=2)}")

    # clean up the new metrics.json
    (output_path / "metrics.json").unlink()
    # clean up the output_path folder
    output_path.rmdir()
