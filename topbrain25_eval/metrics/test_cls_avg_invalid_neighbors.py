"""
run the tests with pytest
"""

import tempfile
from pathlib import Path

import cls_avg_invalid_neighbors
import pytest
import SimpleITK as sitk
from cls_avg_invalid_neighbors import (
    invalid_neighbors_all_classes,
    invalid_neighbors_single_label,
    script_dir,
)
from topbrain25_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

##############################################################
#   ________________________________
# < 5. Tests for invalid neighbors >
#   --------------------------------
#          \   ^__^
#           \  (oo)\_______
#              (__)\       )\/\\
#                  ||----w |
#                  ||     ||
##############################################################

TESTDIR_3D = Path("test_assets/seg_metrics/3D")


def test_invalid_neighbors_single_label():
    # images are NOT used by invalid_neighbors_single_label!

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=script_dir) as f:
        # temp json to simulate pred neighbor json
        f.write('{"1": [314, 2], "2": [123, 321], "3": [1, 16, 17], "4":[]}')
        f.flush()

        # Simulate the global variable json_path being set in the module
        cls_avg_invalid_neighbors.gt_neighbor_json_path = (
            script_dir / "valid_neighbors_ta36.json"
        )
        cls_avg_invalid_neighbors.pred_neighbor_json_path = f.name

        num_invalid_neighbors = invalid_neighbors_single_label(
            gt=None, pred=None, label=1
        )
        assert num_invalid_neighbors == 1

        num_invalid_neighbors = invalid_neighbors_single_label(
            gt=None, pred=None, label=2
        )
        assert num_invalid_neighbors == 2

        num_invalid_neighbors = invalid_neighbors_single_label(
            gt=None, pred=None, label=3
        )
        assert num_invalid_neighbors == 2

        num_invalid_neighbors = invalid_neighbors_single_label(
            gt=None, pred=None, label=4
        )
        assert num_invalid_neighbors == 0

        # missing labels will raise KeyError
        with pytest.raises(KeyError):
            invalid_neighbors_single_label(gt=None, pred=None, label=42)


def test_invalid_neighbors_all_classes_BG_images():
    # two blank all background images
    gt = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    pred = sitk.Image([3, 3, 3], sitk.sitkUInt8)

    invalid_neighbors_dict = invalid_neighbors_all_classes(gt=gt, pred=pred)
    assert invalid_neighbors_dict == {
        "ClsAvgNbErr": {"label": "ClsAvgNbErr", "NbErr": 0},
    }


def test_invalid_neighbors_all_classes_absent_pred():
    """
    Since gt image is not used to compute the invalid neighbors and instead we use
    rule-based set of valid neighbors,
    this means if the pred is absent for that label, no NbErr for that label!
    """

    # image1 is blank, image2 has labels
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    for i in range(3):
        image2[i, i, i] = i + 30

    # depending which image is pred, the NbErr is different!
    invalid_neighbors_dict = invalid_neighbors_all_classes(gt=image1, pred=image2)
    assert invalid_neighbors_dict == {
        "30": {"label": "L-PICA", "NbErr": 1},
        "31": {"label": "R-AChA", "NbErr": 2},
        "32": {"label": "L-AChA", "NbErr": 1},
        "ClsAvgNbErr": {"label": "ClsAvgNbErr", "NbErr": 4 / 3},
    }

    # if pred is blank
    invalid_neighbors_dict = invalid_neighbors_all_classes(gt=image2, pred=image1)
    assert invalid_neighbors_dict == {
        "ClsAvgNbErr": {"label": "ClsAvgNbErr", "NbErr": 0},
    }


def test_invalid_neighbors_all_classes_isolated_blob():
    """
    test for when some classes have no neighbors,
    should proceed with the neigbor check without KeyError: 'X'!
    """

    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    # Assign label values to corner voxels
    image2[0, 0, 0] = 1  # Corner 1
    # image2[0, 0, 2] = 40  # Corner 2

    # saved pred_neighbors.json is cannot be {}!
    invalid_neighbors_dict = invalid_neighbors_all_classes(gt=image2, pred=image2)
    assert invalid_neighbors_dict == {
        "1": {"label": "BA", "NbErr": 0},
        # "40": {"label": "SSS", "NbErr": 0},
        "ClsAvgNbErr": {"label": "ClsAvgNbErr", "NbErr": 0},
    }


def test_invalid_neighbors_all_classes_twoIslands():
    """
    reuse test_twoIslands() from test_cls_avg_b0.py
    """
    pred_path = TESTDIR_3D / "shape_3x4x2_3D_twoIslands.nii.gz"

    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    invalid_neighbors_dict = invalid_neighbors_all_classes(gt=pred_img, pred=pred_img)
    assert invalid_neighbors_dict == {
        "1": {"label": "BA", "NbErr": 0},
        "ClsAvgNbErr": {"label": "ClsAvgNbErr", "NbErr": 0},
    }


def test_invalid_neighbors_all_classes_detection_label8910():
    """
    re-use test cases from test_detection_sideroad_labels_ThresholdIoU()
    detection_label8910_{gt,pred}_squareL4

    --> invalid_neighbors_single_label() for label-8
    gt_neighbors = [2, 4, 25, 31]
    pred_neighbors = [9]
    Elements only in pred: {9}
    num_invalid_neighbors = 1

    --> invalid_neighbors_single_label() for label-9
    gt_neighbors = [3, 6, 26, 32]
    pred_neighbors = [8, 10]
    Elements only in pred: {8, 10}
    num_invalid_neighbors = 2

    --> invalid_neighbors_single_label() for label-10
    gt_neighbors = [11, 12, 15]
    pred_neighbors = [9, 15]
    Elements only in pred: {9}
    num_invalid_neighbors = 1

    --> invalid_neighbors_single_label() for label-15
    gt_neighbors = [10, 11, 12, 13, 14, 16]
    pred_neighbors = [10]
    Elements only in pred: set()
    num_invalid_neighbors = 0
    """

    # NOTE: gt image is not used!
    gt_path = TESTDIR_3D / "detection_label8910_gt_squareL4.nii.gz"
    pred_path = TESTDIR_3D / "detection_label8910_pred_squareL4.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    invalid_neighbors_dict = invalid_neighbors_all_classes(gt=gt_img, pred=pred_img)

    assert invalid_neighbors_dict == {
        "8": {"label": "R-Pcom", "NbErr": 1},
        "9": {"label": "L-Pcom", "NbErr": 2},
        "10": {"label": "Acom", "NbErr": 1},
        "15": {"label": "3rd-A2", "NbErr": 0},
        "ClsAvgNbErr": {"label": "ClsAvgNbErr", "NbErr": 1.0},
    }


# def test_invalid_neighbors_all_classes_detection_label34373839():
#     """
#     re-use test cases from test_detection_sideroad_labels_ThresholdIoU_CT()
#     4 planes of label-34, 37, 38, 39 CT

#     --> invalid_neighbors_single_label() for label-34
#     gt_neighbors = []
#     pred_neighbors = [37]
#     Elements only in pred: {37}
#     num_invalid_neighbors = 1

#     --> invalid_neighbors_single_label() for label-37
#     gt_neighbors = [21, 22, 35, 38, 39]
#     pred_neighbors = [34, 38]
#     Elements only in pred: {34}
#     num_invalid_neighbors = 1

#     --> invalid_neighbors_single_label() for label-38
#     gt_neighbors = [2, 5, 21, 25, 35, 37]
#     pred_neighbors = [37, 39]
#     Elements only in pred: {39}
#     num_invalid_neighbors = 1

#     --> invalid_neighbors_single_label() for label-39
#     gt_neighbors = [3, 7, 22, 26, 35, 37]
#     pred_neighbors = [38]
#     Elements only in pred: {38}
#     num_invalid_neighbors = 1
#     """

#     # NOTE: gt image is not used!
#     gt_path = TESTDIR_3D / "detection_label8910_gt_squareL4_CT.nii.gz"
#     pred_path = TESTDIR_3D / "detection_label8910_pred_squareL4_CT.nii.gz"

#     gt_img, _ = load_image_and_array_as_uint8(gt_path)
#     pred_img, _ = load_image_and_array_as_uint8(pred_path)

#     invalid_neighbors_dict = invalid_neighbors_all_classes(gt=gt_img, pred=pred_img)

#     assert invalid_neighbors_dict == {
#         "34": {"label": "L-OA", "NbErr": 1},
#         "37": {"label": "ICVs", "NbErr": 1},
#         "38": {"label": "R-BVR", "NbErr": 1},
#         "39": {"label": "L-BVR", "NbErr": 1},
#         "ClsAvgNbErr": {"label": "ClsAvgNbErr", "NbErr": 1.0},
#     }


# def test_invalid_neighbors_all_classes_detection_label33344142():
#     """
#     re-use test cases from test_detection_sideroad_labels_ThresholdIoU_MR()
#     4 planes of label-33, 34, 41, 42 MR

#     --> invalid_neighbors_single_label() for label-33
#     gt_neighbors = [4]
#     pred_neighbors = [34]
#     Elements only in pred: {34}
#     num_invalid_neighbors = 1

#     --> invalid_neighbors_single_label() for label-34
#     gt_neighbors = [6]
#     pred_neighbors = [33, 41]
#     Elements only in pred: {33, 41}
#     num_invalid_neighbors = 2

#     --> invalid_neighbors_single_label() for label-41
#     gt_neighbors = [35, 39]
#     pred_neighbors = [34, 42]
#     Elements only in pred: {34, 42}
#     num_invalid_neighbors = 2

#     --> invalid_neighbors_single_label() for label-42
#     gt_neighbors = [36, 40]
#     pred_neighbors = [41]
#     Elements only in pred: {41}
#     num_invalid_neighbors = 1
#     """

#     # NOTE: gt image is not used!
#     gt_path = TESTDIR_3D / "detection_label8910_gt_squareL4_MR.nii.gz"
#     pred_path = TESTDIR_3D / "detection_label8910_pred_squareL4_MR.nii.gz"

#     gt_img, _ = load_image_and_array_as_uint8(gt_path)
#     pred_img, _ = load_image_and_array_as_uint8(pred_path)

#     invalid_neighbors_dict = invalid_neighbors_all_classes(gt=gt_img, pred=pred_img)

#     assert invalid_neighbors_dict == {
#         "33": {"label": "R-OA", "NbErr": 1},
#         "34": {"label": "L-OA", "NbErr": 2},
#         "41": {"label": "R-MMA", "NbErr": 2},
#         "42": {"label": "L-MMA", "NbErr": 1},
#         "ClsAvgNbErr": {"label": "ClsAvgNbErr", "NbErr": 1.5},
#     }


def test_invalid_neighbors_all_classes_023LPS_ICA_PCA_flipped():
    """
    re-use imgs from test_DiceCoefficient_topcow023mr()
    LPS_ICA_PCA_flipped.nii.gz contains labels 1,2,3,4,6 without invalid neighbors
    """
    testdir = Path("test_assets/seg_metrics/topcow_roi")

    gt, _ = load_image_and_array_as_uint8(testdir / "topcow_mr_roi_023.nii.gz", True)
    pred, _ = load_image_and_array_as_uint8(
        testdir / "LPS_ICA_PCA_flipped.nii.gz", True
    )

    invalid_neighbors_dict = invalid_neighbors_all_classes(gt=gt, pred=pred)

    assert invalid_neighbors_dict == {
        "1": {"label": "BA", "NbErr": 0},
        "2": {"label": "R-P1P2", "NbErr": 0},
        "3": {"label": "L-P1P2", "NbErr": 0},
        "4": {"label": "R-ICA-C6-C7", "NbErr": 0},
        "6": {"label": "L-ICA-C6-C7", "NbErr": 0},
        "ClsAvgNbErr": {"label": "ClsAvgNbErr", "NbErr": 0},
    }
