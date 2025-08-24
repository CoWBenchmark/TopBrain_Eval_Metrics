from pathlib import Path

import SimpleITK as sitk

from topbrain25_eval.constants import TRACK
from topbrain25_eval.score_case_task_1_seg import score_case_task_1_seg

TESTDIR = Path("./test_assets")


def test_score_case_task_1_seg_ct():
    """
    score_case_task_1_seg() should be the same as
    test_e2e_TopCoWEvaluation_Task_1_Seg_no_crop's
    "case": self._case_results.to_dict() part,
    except the pred_fname and gt_fname path fields
    """
    test_dict = {"UZH": "Best #1"}
    gt = sitk.ReadImage(
        TESTDIR / "task_1_seg_ground-truth" / "shape_8x8x8_3D_8Cubes_gt.nii.gz"
    )
    pred = sitk.ReadImage(
        TESTDIR / "task_1_seg_predictions" / "shape_8x8x8_3D_8Cubes_pred.mha"
    )

    # should only mutate test_dict
    score_case_task_1_seg(track=TRACK.CT, gt=gt, pred=pred, metrics_dict=test_dict)

    # test_dict object still can be mutated
    test_dict["NUS"] = 2022

    print("test_dict =", test_dict)

    # thus the original parts of the dict are kept
    assert test_dict == {
        "UZH": "Best #1",
        "Dice_BA": 1.0,
        "Dice_R-P1P2": 0.9333333333333333,
        "Dice_L-P1P2": 1.0,
        "Dice_R-ICA": 0,
        "Dice_R-M1": 1.0,
        "Dice_L-ICA": 0.8571428571428571,
        "Dice_L-M1": 1.0,
        "Dice_R-Pcom": 0.8571428571428571,
        "Dice_ClsAvgDice": 0.8309523809523809,
        "Dice_MergedBin": 0.8869565217391304,
        "clDice_BA": 0,
        "clDice_R-P1P2": 0.0,
        "clDice_L-P1P2": 0,
        "clDice_R-ICA": 0,
        "clDice_R-M1": 0,
        "clDice_L-ICA": 0.0,
        "clDice_L-M1": 0,
        "clDice_R-Pcom": 0.0,
        "clDice_ClsAvgclDice": 0.0,
        "clDice_MergedBin": 0.0,
        "B0err_BA": 0,
        "B0err_R-P1P2": 0,
        "B0err_L-P1P2": 0,
        "B0err_R-ICA": 1,
        "B0err_R-M1": 0,
        "B0err_L-ICA": 0,
        "B0err_L-M1": 0,
        "B0err_R-Pcom": 0,
        "B0err_ClsAvgB0err": 0.125,
        "B0err_MergedBin": 0,
        "HD_BA": 0.0,
        "HD95_BA": 0.0,
        "HD_R-P1P2": 0.0,
        "HD95_R-P1P2": 0.0,
        "HD_L-P1P2": 0.0,
        "HD95_L-P1P2": 0.0,
        "HD_R-ICA": 290,
        "HD95_R-ICA": 290,
        "HD_R-M1": 0.0,
        "HD95_R-M1": 0.0,
        "HD_L-ICA": 1.0,
        "HD95_L-ICA": 1.0,
        "HD_L-M1": 0.0,
        "HD95_L-M1": 0.0,
        "HD_R-Pcom": 1.0,
        "HD95_R-Pcom": 1.0,
        "HD95_ClsAvgHD95": 36.5,
        "HD_ClsAvgHD": 36.5,
        "HD_MergedBin": 4.0,
        "HD95_MergedBin": 2.0,
        "NbErr_BA": 4,
        "NbErr_R-P1P2": 4,
        "NbErr_L-P1P2": 4,
        "NbErr_R-M1": 6,
        "NbErr_L-ICA": 4,
        "NbErr_L-M1": 5,
        "NbErr_R-Pcom": 5,
        "NbErr_ClsAvgNbErr": 4.571428571428571,
        "all_detection_dicts": {
            "8": {"label": "R-Pcom", "Detection": "TP"},
            "9": {"label": "L-Pcom", "Detection": "TN"},
            "10": {"label": "Acom", "Detection": "TN"},
            "15": {"label": "3rd-A2", "Detection": "TN"},
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
            "37": {"label": "ICVs", "Detection": "TN"},
            "38": {"label": "R-BVR", "Detection": "TN"},
            "39": {"label": "L-BVR", "Detection": "TN"},
        },
        "NUS": 2022,
    }


def test_score_case_task_1_seg_mr():
    """
    re-use from test_clDice_all_classes()
    """
    test_dict = {"UZH": "best in CH!"}

    gt = sitk.ReadImage(
        TESTDIR / "seg_metrics/3D/shape_5x6x4_multiclass_clDice_gt.nii.gz"
    )
    pred = sitk.ReadImage(
        TESTDIR / "seg_metrics/3D/shape_5x6x4_multiclass_clDice_pred1.nii.gz"
    )

    # should only mutate test_dict
    score_case_task_1_seg(track=TRACK.MR, gt=gt, pred=pred, metrics_dict=test_dict)

    # test_dict object still can be mutated
    test_dict["Marrakesh"] = "2024"

    print("test_dict =", test_dict)

    # thus the original parts of the dict are kept
    assert test_dict == {
        "UZH": "best in CH!",
        "Dice_L-STA": 0.9333333333333333,
        "Dice_R-MaxA": 0.5263157894736842,
        "Dice_L-MaxA": 0.6,
        "Dice_ClsAvgDice": 0.6865497076023392,
        "Dice_MergedBin": 0.6666666666666666,
        "clDice_L-STA": 0.8571428571428571,
        "clDice_R-MaxA": 0.8571428571428571,
        "clDice_L-MaxA": 1.0,
        "clDice_ClsAvgclDice": 0.9047619047619048,
        "clDice_MergedBin": 0.8333333333333333,
        "B0err_L-STA": 0,
        "B0err_R-MaxA": 0,
        "B0err_L-MaxA": 0,
        "B0err_ClsAvgB0err": 0.0,
        "B0err_MergedBin": 0,
        "HD_L-STA": 1.0,
        "HD95_L-STA": 0.6499999999999995,
        "HD_R-MaxA": 3.0,
        "HD95_R-MaxA": 3.0,
        "HD_L-MaxA": 1.0,
        "HD95_L-MaxA": 1.0,
        "HD95_ClsAvgHD95": 1.5499999999999998,
        "HD_ClsAvgHD": 1.6666666666666667,
        "HD_MergedBin": 1.4142135381698608,
        "HD95_MergedBin": 1.1035533845424652,
        "NbErr_L-STA": 1,
        "NbErr_R-MaxA": 2,
        "NbErr_L-MaxA": 1,
        "NbErr_ClsAvgNbErr": 1.3333333333333333,
        "all_detection_dicts": {
            "8": {"label": "R-Pcom", "Detection": "TN"},
            "9": {"label": "L-Pcom", "Detection": "TN"},
            "10": {"label": "Acom", "Detection": "TN"},
            "15": {"label": "3rd-A2", "Detection": "TN"},
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
            "41": {"label": "R-MMA", "Detection": "TN"},
            "42": {"label": "L-MMA", "Detection": "TN"},
        },
        "Marrakesh": "2024",
    }
