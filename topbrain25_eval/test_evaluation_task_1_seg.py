"""
End-to-end test for the entire evaluation pipeline
"""

import json
from pathlib import Path

from topbrain25_eval.constants import TRACK
from topbrain25_eval.evaluation import TopBrainEvaluation

TESTDIR = Path("test_assets/")


def test_e2e_TopBrainEvaluation_Task_1_Seg_CT():
    """
    under test_assets/, there are two files for
    test-predictions, test-gt
        "gt_fname": {
            "0": "shape_5x7x9_3D_1donut.nii.gz",
            "1": "shape_8x8x8_3D_8Cubes_gt.nii.gz"
        },
        "pred_fname": {
            "0": "shape_5x7x9_3D_1donut_multiclass.mha",
            "1": "shape_8x8x8_3D_8Cubes_pred.mha"
        }
        ...
    in:
    - ./test_assets/task_1_seg_predictions
    - ./test_assets/task_1_seg_ground-truth

    There is also an expected metrics.json output in:
    - ./test_assets/task_1_seg_output

        num_input_pred = 2
        num_ground_truth = 2
        task_1_seg_predictions/
        ├── shape_5x7x9_3D_1donut_multiclass.mha
        └── shape_8x8x8_3D_8Cubes_pred.mha
        task_1_seg_ground-truth/
        ├── shape_5x7x9_3D_1donut.nii.gz
        └── shape_8x8x8_3D_8Cubes_gt.nii.gz


    This test runs TopBrainEvaluation().evaluate() for Head-Angio-Segmentation
    and compares the metrics.json with the expected metrics.json.

    This is the final e2e test that the whole pipeline must pass

    NOTE: Gotcha for label-2 HD/HD95 in shape_8x8x8_3D_8Cubes_pred
    the HD/HD95 is 0 even though gt and pred are different
    (the segmentation surface contour is the same...).

    This is similar to example from Fig 59 of
    Common Limitations of Image Processing Metrics: A Picture Story
    with hole inside Pred 2 bigger hole.
    see test_hd95_single_label_Fig59_hole() from test_cls_avg_hd95.py
    """

    track = TRACK.CT

    expected_num_cases = 2

    # folder prefix to differentiate the tasks
    prefix = "task_1_seg_"

    # output_path for clean up
    output_path = Path(f"{prefix}output_test_e2e_TopBrainEvaluation_CT/")

    evalRun = TopBrainEvaluation(
        track,
        expected_num_cases,
        predictions_path=TESTDIR / f"{prefix}predictions/",
        ground_truth_path=TESTDIR / f"{prefix}ground-truth/",
        output_path=output_path,
    )

    # run the evaluation
    evalRun.evaluate()

    # read the saved metrics.json
    with open(output_path / "metrics.json") as f:
        generated_metrics_json = json.load(f)

    # some sanity checks
    # file is sorted by filename, so 0th file is shape_5x7x9_3D_1donut

    ################# 0th-file #################
    f_idx = "0"
    # Dice same as test_multi_class_donut() in test_cls_avg_dice.py
    assert round(generated_metrics_json["case"]["Dice_BA"][f_idx], 2) == 0.75
    assert generated_metrics_json["case"]["Dice_L-ICA"][f_idx] == 0
    assert (
        round(generated_metrics_json["case"]["Dice_ClsAvgDice"][f_idx], 3) == 0.375
    )  # 0.75/2
    assert generated_metrics_json["case"]["Dice_MergedBin"][f_idx] == 1
    # B0 same as test_multi_class_donut() in test_cls_avg_b0.py
    assert generated_metrics_json["case"]["B0err_BA"][f_idx] == 0
    assert generated_metrics_json["case"]["B0err_L-ICA"][f_idx] == 1
    assert generated_metrics_json["case"]["B0err_ClsAvgB0err"][f_idx] == 0.5
    assert generated_metrics_json["case"]["B0err_MergedBin"][f_idx] == 0
    # NbErr both labels error -> avg 1
    assert generated_metrics_json["case"]["NbErr_ClsAvgNbErr"][f_idx] == 1
    # detection Not applicable -> all TN
    assert (
        generated_metrics_json["case"]["all_detection_dicts"][f_idx]["8"]["Detection"]
        == "TN"
    )

    ################# 1th-file #################
    f_idx = "1"
    # Dice same as test_dice_dict_e2e() in test_cls_avg_dice.py
    assert generated_metrics_json["case"]["Dice_BA"][f_idx] == 1
    assert round(generated_metrics_json["case"]["Dice_R-P1P2"][f_idx], 2) == 0.93
    assert generated_metrics_json["case"]["Dice_L-P1P2"][f_idx] == 1
    assert round(generated_metrics_json["case"]["Dice_ClsAvgDice"][f_idx], 2) == 0.83
    assert round(generated_metrics_json["case"]["Dice_MergedBin"][f_idx], 2) == 0.89
    # B0 same as test_betti_num_err_dict_e2e() in test_cls_avg_b0.py
    assert generated_metrics_json["case"]["B0err_ClsAvgB0err"][f_idx] == 1 / 8
    # NbErr
    # -> label-1 {5,6,7,8}
    assert generated_metrics_json["case"]["NbErr_BA"][f_idx] == 4
    # -> label-2 {3,5,6,7}
    assert generated_metrics_json["case"]["NbErr_R-P1P2"][f_idx] == 4
    # -> label-3 {2,5,7,8}
    assert generated_metrics_json["case"]["NbErr_L-P1P2"][f_idx] == 4
    # -> label-5 {1,2,3,6,7,8}
    assert generated_metrics_json["case"]["NbErr_R-M1"][f_idx] == 6
    # -> label-6 {1,2,5,8}
    assert generated_metrics_json["case"]["NbErr_L-ICA"][f_idx] == 4
    # -> label-7 {1,2,3,5,8}
    assert generated_metrics_json["case"]["NbErr_L-M1"][f_idx] == 5
    # -> label-8 {1,3,5,6,7}
    assert generated_metrics_json["case"]["NbErr_R-Pcom"][f_idx] == 5
    assert generated_metrics_json["case"]["NbErr_ClsAvgNbErr"][f_idx] == 32 / 7  # ~4.57

    # average of above sanity checks
    assert round(generated_metrics_json["aggregates"]["Dice_BA"]["mean"], 3) == 1.75 / 2
    assert round(
        generated_metrics_json["aggregates"]["Dice_L-ICA"]["mean"], 2
    ) == round(0.8571 / 2, 2)
    assert generated_metrics_json["aggregates"]["Dice_L-P1P2"]["mean"] == 1
    assert round(
        generated_metrics_json["aggregates"]["Dice_ClsAvgDice"]["mean"], 2
    ) == round((0.375 + 0.83) / 2, 2)  # ~0.6
    assert generated_metrics_json["aggregates"]["B0err_R-ICA"]["mean"] == 1
    assert generated_metrics_json["aggregates"]["B0err_L-ICA"]["mean"] == 1 / 2
    assert (
        generated_metrics_json["aggregates"]["B0err_ClsAvgB0err"]["mean"]
        == (0.5 + 0.125) / 2
    )
    assert round(
        generated_metrics_json["aggregates"]["NbErr_ClsAvgNbErr"]["mean"], 2
    ) == round((1 + 4.57) / 2, 2)  # ~2.7857

    # compare the saved and expected metrics.json

    with open(TESTDIR / f"{prefix}output/expected_e2e_test_ct_metrics.json") as f:
        expected_metrics_json = json.load(f)

    assert expected_metrics_json == generated_metrics_json

    print(f"expected_metrics_json =\n{json.dumps(expected_metrics_json, indent=2)}")
    print(f"generated_metrics_json =\n{json.dumps(generated_metrics_json, indent=2)}")

    # clean up the new metrics.json
    (output_path / "metrics.json").unlink()
    # clean up the output_path folder
    output_path.rmdir()


def test_e2e_TopBrainEvaluation_Task_1_Seg_MR():
    """
    same as test_e2e_TopBrainEvaluation_Task_1_Seg_CT but for MR
    """
    # NOTE: changing the track to CT from MR
    track = TRACK.MR

    expected_num_cases = 2

    # folder prefix to differentiate the tasks
    prefix = "task_1_seg_"

    # output_path for clean up
    output_path = Path(f"{prefix}output_test_e2e_TopBrainEvaluation_MR/")

    evalRun = TopBrainEvaluation(
        track,
        expected_num_cases,
        predictions_path=TESTDIR / f"{prefix}predictions/",
        ground_truth_path=TESTDIR / f"{prefix}ground-truth/",
        output_path=output_path,
    )

    # run the evaluation
    evalRun.evaluate()

    # compare the two metrics.json

    with open(TESTDIR / f"{prefix}output/expected_e2e_test_mr_metrics.json") as f:
        expected_metrics_json = json.load(f)

    with open(output_path / "metrics.json") as f:
        generated_metrics_json = json.load(f)

    assert expected_metrics_json == generated_metrics_json

    print(f"expected_metrics_json =\n{json.dumps(expected_metrics_json, indent=2)}")
    print(f"generated_metrics_json =\n{json.dumps(generated_metrics_json, indent=2)}")

    # clean up the new metrics.json
    (output_path / "metrics.json").unlink()
    # clean up the output_path folder
    output_path.rmdir()
