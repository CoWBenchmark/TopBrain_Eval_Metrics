"""
run the tests with pytest
"""

from pathlib import Path

import SimpleITK as sitk
from detection_sideroad_labels import (
    detection_sideroad_labels,
    detection_single_label,
    iou_single_label,
)
from topbrain25_eval.constants import TRACK
from topbrain25_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

##############################################################
#   __________________________________________
# < 6. Tests for detection of side-road vessel >
#   ------------------------------------------
#          \   ^__^
#           \  (oo)\_______
#              (__)\       )\/\\
#                  ||----w |
#                  ||     ||
##############################################################

TESTDIR_2D = Path("test_assets/seg_metrics/2D")
TESTDIR_3D = Path("test_assets/seg_metrics/3D")


###########################################################
# tests for iou_single_label()


def test_iou_single_label_Fig10():
    """
    example from Fig 10 of
    Common Limitations of Image Processing Metrics: A Picture Story

    Fig10 caption gives IoU = 0.8
    """
    gt_path = TESTDIR_2D / "shape_11x11_2D_Fig10_bin_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_11x11_2D_Fig10_bin_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    iou_score = iou_single_label(gt=gt_img, pred=pred_img, label=1)
    # iou is actually 0.757, round to 0.8
    assert round(iou_score, 1) == round(0.757, 1) == 0.8


def test_iou_single_label_Fig50_small_multiclass():
    """
    example from Fig 50 of
    Common Limitations of Image Processing Metrics: A Picture Story

    small structure from bottom row

    Fig50 caption gives IoU = 0.67 for Pred 1 (label-10)
                        IoU = 0.33 for Pred 2 (label-15)
    """
    gt_path = TESTDIR_3D / "shape_2x2x2_3D_Fig50_label15_10_gt.nii.gz"
    pred_path = TESTDIR_3D / "shape_2x2x2_3D_Fig50_label15_10_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # for label-1, there is no mask for computation
    # IoU should be 0
    label = 1
    iou_score = iou_single_label(gt=gt_img, pred=pred_img, label=label)
    assert iou_score == 0

    # IoU = 0.67 for Pred 1 (label-10)
    label = 10
    iou_score = iou_single_label(gt=gt_img, pred=pred_img, label=label)
    assert iou_score == 0.6666666666666666

    # IoU = 0.33 for Pred 2 (label-15)
    label = 15
    iou_score = iou_single_label(gt=gt_img, pred=pred_img, label=label)
    assert iou_score == 0.3333333333333333


def test_iou_single_label_Fig54_5class():
    """
    example from Fig 54 of
    Common Limitations of Image Processing Metrics: A Picture Story

    5 planes for Pred 1-5 with label 1-5
    """
    gt_path = TESTDIR_3D / "shape_11x11x5_3D_Fig54_label1-5_gt.nii.gz"
    pred_path = TESTDIR_3D / "shape_11x11x5_3D_Fig54_label1-5_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # NOTE: all IoU = 0.4!

    # Pred 1 IOU = 0.4
    label = 1
    iou_score = iou_single_label(gt=gt_img, pred=pred_img, label=label)
    assert iou_score == 0.42857142857142855

    # Pred 2 IOU = 0.4
    label = 2
    iou_score = iou_single_label(gt=gt_img, pred=pred_img, label=label)
    assert iou_score == 0.42857142857142855

    # Pred 3 IOU = 0.4
    label = 3
    iou_score = iou_single_label(gt=gt_img, pred=pred_img, label=label)
    assert iou_score == 0.42857142857142855

    # Pred 4 IOU = 0.4
    label = 4
    iou_score = iou_single_label(gt=gt_img, pred=pred_img, label=label)
    assert iou_score == 0.42857142857142855

    # Pred 5 IOU = 0.4
    label = 5
    iou_score = iou_single_label(gt=gt_img, pred=pred_img, label=label)
    assert iou_score == 0.42857142857142855


def test_iou_single_label_blank_slices():
    """
    iou_single_label() but with blank slices
    """
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)

    # slice 0 image1 blank, image2 has one label-7
    image2[:, :, 0] = 7
    # slice 1 image1 has two label-8, image2 blank
    image1[2, 0, 1] = 8
    image1[0, 2, 1] = 8
    # slice 2 both blank

    print("image1:")
    print(sitk.GetArrayViewFromImage(image1))

    print("image2:")
    print(sitk.GetArrayViewFromImage(image2))

    # label-7
    label = 7
    iou = iou_single_label(gt=image1, pred=image2, label=label)
    assert iou == 0

    # label-8
    label = 8
    iou = iou_single_label(gt=image1, pred=image2, label=label)
    assert iou == 0

    # label-9 (not in either gt or pred)
    label = 9
    iou = iou_single_label(gt=image1, pred=image2, label=label)
    assert iou == 0


def test_iou_coefficient_single_label_3D_voxel_spacing():
    """
    with simple 3x3x3 sitk.Image, test that
    iou_single_label calculation DONT use spacing information
    """
    label = 42

    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)

    image1[0, :, :] = label

    image2[:, 2, :] = label

    print("image1:")
    print(sitk.GetArrayViewFromImage(image1))

    print("image2:")
    print(sitk.GetArrayViewFromImage(image2))

    # without SetSpacing
    without_spacing = iou_single_label(
        gt=image1,
        pred=image2,
        label=label,
    )
    # w/o spacing, IoU = 0.2
    assert without_spacing == 3 / (9 + 6)

    # Set Spacing
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    # image1 and image2 even have different spacings!
    image1.SetSpacing((1, 3, 5))
    image2.SetSpacing((3, 7, 9))

    image1[0, :, :] = label

    image2[:, 2, :] = label

    print("image1:")
    print(sitk.GetArrayViewFromImage(image1))

    print("image2:")
    print(sitk.GetArrayViewFromImage(image2))

    with_spacing = iou_single_label(
        gt=image1,
        pred=image2,
        label=label,
    )
    # NOTE: with spacing, even different spacings!
    # IoU is still 0.2
    assert with_spacing == without_spacing


###########################################################
# tests for detection_single_label()


def test_detection_single_label_ThresholdIoU():
    """
    4 planes of label-8, 9, 10, 15
    Label-8  GT 4x4      Pred 4 voxels
    Label-9  GT 4x4      Pred 3 voxels
    Label-10 GT 4x4      Pred 5 voxels
    Label-15 GT missing  Pred 1 voxel

    # label-8 IoU = 0.25 -> TP
    "8": {"label": "R-Pcom", "Detection": "TP"},
    # label-9 IoU < 0.25 -> FN
    "9": {"label": "L-Pcom", "Detection": "FN"},
    # label-10 IoU > 0.25 -> TP
    "10": {"label": "Acom", "Detection": "TP"},
    # label-15 GT missing, pred not -> FP
    "15": {"label": "3rd-A2", "Detection": "FP"},
    """
    gt_path = TESTDIR_3D / "detection_label8910_gt_squareL4.nii.gz"
    pred_path = TESTDIR_3D / "detection_label8910_pred_squareL4.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # label-8 IoU = 0.25 -> TP
    label = 8
    detection = detection_single_label(gt=gt_img, pred=pred_img, label=label)
    assert detection == "TP"

    # label-9 IoU < 0.25 -> FN
    label = 9
    detection = detection_single_label(gt=gt_img, pred=pred_img, label=label)
    assert detection == "FN"

    # label-10 IoU > 0.25 -> TP
    label = 10
    detection = detection_single_label(gt=gt_img, pred=pred_img, label=label)
    assert detection == "TP"

    # label-15 GT missing, pred not -> FP
    label = 15
    detection = detection_single_label(gt=gt_img, pred=pred_img, label=label)
    assert detection == "FP"

    # label-42 GT missing, Pred also missing -> TN
    label = 42
    detection = detection_single_label(gt=gt_img, pred=pred_img, label=label)
    assert detection == "TN"


###########################################################
# tests for detection_sideroad_labels()


def test_detection_sideroad_labels_Fig50_small_multiclass():
    """
    example from Fig 50 of
    Common Limitations of Image Processing Metrics: A Picture Story

    small structure from bottom row

    Fig50 caption gives IoU = 0.67 for Pred 1 (label-10)
                        IoU = 0.33 for Pred 2 (label-15)
    """
    gt_path = TESTDIR_3D / "shape_2x2x2_3D_Fig50_label15_10_gt.nii.gz"
    pred_path = TESTDIR_3D / "shape_2x2x2_3D_Fig50_label15_10_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    expected_out = {
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

    detection_dict_ct = detection_sideroad_labels(
        track=TRACK.CT, gt=gt_img, pred=pred_img
    )

    # label 8 and 9, nothing in GT and Pred, thus TN
    assert detection_dict_ct == expected_out | {
        # only in CT
        "37": {"label": "ICVs", "Detection": "TN"},
        "38": {"label": "R-BVR", "Detection": "TN"},
        "39": {"label": "L-BVR", "Detection": "TN"},
    }

    detection_dict_mr = detection_sideroad_labels(
        track=TRACK.MR, gt=gt_img, pred=pred_img
    )

    # label 8 and 9, nothing in GT and Pred, thus TN
    assert detection_dict_mr == expected_out | {
        # only in MR
        "41": {"label": "R-MMA", "Detection": "TN"},
        "42": {"label": "L-MMA", "Detection": "TN"},
    }


def test_detection_sideroad_labels_ThresholdIoU():
    """
    4 planes of label-8, 9, 10, 15
    Label-8  GT 4x4      Pred 4 voxels
    Label-9  GT 4x4      Pred 3 voxels
    Label-10 GT 4x4      Pred 5 voxels
    Label-15 GT missing  Pred 1 voxel
    """
    gt_path = TESTDIR_3D / "detection_label8910_gt_squareL4.nii.gz"
    pred_path = TESTDIR_3D / "detection_label8910_pred_squareL4.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    expected_out = {
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

    detection_dict_ct = detection_sideroad_labels(
        track=TRACK.CT, gt=gt_img, pred=pred_img
    )

    assert detection_dict_ct == expected_out | {
        # only in CT
        "37": {"label": "ICVs", "Detection": "TN"},
        "38": {"label": "R-BVR", "Detection": "TN"},
        "39": {"label": "L-BVR", "Detection": "TN"},
    }

    detection_dict_mr = detection_sideroad_labels(
        track=TRACK.MR, gt=gt_img, pred=pred_img
    )

    assert detection_dict_mr == expected_out | {
        # only in MR
        "41": {"label": "R-MMA", "Detection": "TN"},
        "42": {"label": "L-MMA", "Detection": "TN"},
    }


def test_detection_sideroad_labels_ThresholdIoU_CT():
    """
    4 planes of label-34, 37, 38, 39
    Label-39  GT 4x4      Pred 4 voxels
    Label-38  GT 4x4      Pred 3 voxels
    Label-37  GT 4x4      Pred 5 voxels
    Label-34  GT missing  Pred 1 voxel
    """
    gt_path = TESTDIR_3D / "detection_label8910_gt_squareL4_CT.nii.gz"
    pred_path = TESTDIR_3D / "detection_label8910_pred_squareL4_CT.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    detection_dict = detection_sideroad_labels(track=TRACK.CT, gt=gt_img, pred=pred_img)

    assert detection_dict == {
        # label-39 IoU = 0.25 -> TP
        "39": {"label": "L-BVR", "Detection": "TP"},
        # label-38 IoU < 0.25 -> FN
        "38": {"label": "R-BVR", "Detection": "FN"},
        # label-37 IoU > 0.25 -> TP
        "37": {"label": "ICVs", "Detection": "TP"},
        # label-34 GT missing, pred not -> FP
        "34": {"label": "L-OA", "Detection": "FP"},
        # remaining all TN
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
    }


def test_detection_sideroad_labels_ThresholdIoU_MR():
    """
    4 planes of label-33, 34, 41, 42
    Label-42  GT 4x4      Pred 4 voxels
    Label-41  GT 4x4      Pred 3 voxels
    Label-34  GT 4x4      Pred 5 voxels
    Label-33  GT missing  Pred 1 voxel
    """
    gt_path = TESTDIR_3D / "detection_label8910_gt_squareL4_MR.nii.gz"
    pred_path = TESTDIR_3D / "detection_label8910_pred_squareL4_MR.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    detection_dict = detection_sideroad_labels(track=TRACK.MR, gt=gt_img, pred=pred_img)

    assert detection_dict == {
        # label-42 IoU = 0.25 -> TP
        "42": {"label": "L-MMA", "Detection": "TP"},
        # label-41 IoU < 0.25 -> FN
        "41": {"label": "R-MMA", "Detection": "FN"},
        # label-34 IoU > 0.25 -> TP
        "34": {"label": "L-OA", "Detection": "TP"},
        # label-33 GT missing, pred not -> FP
        "33": {"label": "R-OA", "Detection": "FP"},
        # remaining all TN
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
    }
