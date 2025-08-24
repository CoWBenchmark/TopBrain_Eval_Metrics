"""
run the tests with pytest
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from cls_avg_clDice import cl_score, clDice, clDice_all_classes, clDice_single_label
from skimage.morphology import skeletonize, skeletonize_3d
from topbrain25_eval.constants import TRACK
from topbrain25_eval.utils.utils_mask import (
    convert_multiclass_to_binary,
    extract_labels,
)
from topbrain25_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

##############################################################
#   ________________________________
# < 2. Tests for cl_score and clDice >
#   --------------------------------
#          \   ^__^
#           \  (oo)\_______
#              (__)\       )\/\\
#                  ||----w |
#                  ||     ||
##############################################################

TESTDIR_2D = Path("test_assets/seg_metrics/2D")
TESTDIR_3D = Path("test_assets/seg_metrics/3D")


def test_cl_score_skeletonize_ellipse():
    """
    from skimage example on skeletonize ellipse
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize

    compare the ellipse with a vertical rod and a horizontal rod
    """
    X, Y = np.ogrid[0:9, 0:9]
    ellipse = (1.0 / 3 * (X - 4) ** 2 + (Y - 4) ** 2 < 3**2).astype(np.uint8)
    # ellipse is:
    #  ([[0, 0, 0, 1, 1, 1, 0, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)
    # and the skeletonize(ellipse) is:
    #  ([[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    ########################################
    # compare ellipse with a vertical rod
    ########################################

    v_rod = np.zeros((9, 9), dtype=np.uint8)
    v_rod[:, 4] = 1
    #   ([[0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=uint8)

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=skeletonize(ellipse), v_image=v_rod)
    assert tprec == (4 / 4)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=skeletonize(v_rod), v_image=ellipse)
    assert tsens == (9 / 9)

    ########################################
    # compare ellipse with a horizontal rod
    ########################################
    h_rod = np.zeros((9, 9), dtype=np.uint8)
    h_rod[4, :] = 1
    # array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=skeletonize(ellipse), v_image=h_rod)
    assert tprec == (1 / 4)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=skeletonize(h_rod), v_image=ellipse)
    assert tsens == (5 / 9)


def test_cl_score_2D_blob():
    """
    6x3 2D with an elongated blob gt and a vertical columnn pred
    this test for cl_score (topology precision & topology sensitivity)
    """
    gt_path = TESTDIR_2D / "shape_6x3_2D_clDice_elong_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_6x3_2D_clDice_elong_pred.nii.gz"

    gt_img, gt_mask = load_image_and_array_as_uint8(gt_path)
    pred_img, pred_mask = load_image_and_array_as_uint8(pred_path)

    # clDice makes use of the skimage skeletonize method
    # see https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#skeletonize
    if len(pred_mask.shape) == 2:
        call_skeletonize = skeletonize
    elif len(pred_mask.shape) == 3:
        call_skeletonize = skeletonize_3d

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=call_skeletonize(pred_mask), v_image=gt_mask)
    assert tprec == (6 / 6)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=call_skeletonize(gt_mask), v_image=pred_mask)
    assert tsens == (4 / 4)

    # clDice = 2 * tprec * tsens / (tprec + tsens)
    assert clDice(v_p_pred=pred_mask, v_l_gt=gt_mask) == 1

    # sanity check how labels are there
    labels = extract_labels(gt_mask, pred_mask)
    print(f"labels = {labels}")

    # the wrapper clDice_single_label() should be the same as clDice() itself
    assert clDice_single_label(gt=gt_img, pred=pred_img, label=1) == 1

    clDice_dict = clDice_all_classes(track=TRACK.CT, gt=gt_img, pred=pred_img)
    assert clDice_dict == {
        "1": {"label": "BA", "clDice": 1.0},
        "ClsAvgclDice": {"label": "ClsAvgclDice", "clDice": 1.0},
        "MergedBin": {"label": "MergedBin", "clDice": 1.0},
    }


def test_cl_score_2D_Tshaped():
    """
    5x5 2D with a T-shaped blob gt and a vertical columnn pred
    this test for cl_score (topology precision & topology sensitivity)
    """
    gt_path = TESTDIR_2D / "shape_5x5_2D_clDice_Tshaped_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_5x5_2D_clDice_Tshaped_pred.nii.gz"

    gt_img, gt_mask = load_image_and_array_as_uint8(gt_path)
    pred_img, pred_mask = load_image_and_array_as_uint8(pred_path)

    # clDice makes use of the skimage skeletonize method
    # see https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#skeletonize
    if len(pred_mask.shape) == 2:
        call_skeletonize = skeletonize
    elif len(pred_mask.shape) == 3:
        call_skeletonize = skeletonize_3d

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=call_skeletonize(pred_mask), v_image=gt_mask)
    assert tprec == (5 / 5)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=call_skeletonize(gt_mask), v_image=pred_mask)
    assert tsens == (3 / 4)

    # clDice = 2 * tprec * tsens / (tprec + tsens)
    assert clDice(v_p_pred=pred_mask, v_l_gt=gt_mask) == (3 / 2) / (7 / 4)
    # ~= 0.85714

    # log all existing labels
    labels = extract_labels(gt_mask, pred_mask)
    print(f"labels = {labels}")
    # --> labels = [1]

    # the wrapper clDice_single_label() should be the same as clDice() itself
    assert clDice_single_label(gt=gt_img, pred=pred_img, label=1) == (3 / 2) / (7 / 4)
    assert clDice_single_label(gt=gt_img, pred=pred_img, label=2) == 0

    # only 1 label
    clDice_dict = clDice_all_classes(track=TRACK.MR, gt=gt_img, pred=pred_img)
    assert clDice_dict == {
        "1": {"label": "BA", "clDice": (3 / 2) / (7 / 4)},
        "ClsAvgclDice": {"label": "ClsAvgclDice", "clDice": (3 / 2) / (7 / 4)},
        "MergedBin": {"label": "MergedBin", "clDice": (3 / 2) / (7 / 4)},
    }

    """
    same as test_cl_score_2D_Tshaped but on multiclass
    """
    # with multiclass labels
    multiclass_gt_path = TESTDIR_2D / "shape_5x5_2D_clDice_Tshaped_multiclass_gt.nii.gz"
    multiclass_pred_path = (
        TESTDIR_2D / "shape_5x5_2D_clDice_Tshaped_multiclass_pred.nii.gz"
    )
    _, multiclass_gt_arr = load_image_and_array_as_uint8(multiclass_gt_path)
    _, multiclass_pred_arr = load_image_and_array_as_uint8(multiclass_pred_path)

    # test_cl_score_2D_Tshaped should match merged multiclass
    assert (
        tprec
        == cl_score(
            s_skeleton=call_skeletonize(
                convert_multiclass_to_binary(multiclass_pred_arr)
            ),
            v_image=convert_multiclass_to_binary(multiclass_gt_arr),
        )
        == (5 / 5)
    )

    assert (
        tsens
        == cl_score(
            s_skeleton=call_skeletonize(
                convert_multiclass_to_binary(multiclass_gt_arr)
            ),
            v_image=convert_multiclass_to_binary(multiclass_pred_arr),
        )
        == (3 / 4)
    )

    assert (
        clDice(v_p_pred=pred_mask, v_l_gt=gt_mask)
        == clDice(v_p_pred=multiclass_pred_arr, v_l_gt=multiclass_gt_arr)
        == ((3 / 2) / (7 / 4))
    )
    # ~= 0.85714

    # log all existing labels
    labels = extract_labels(multiclass_gt_arr, multiclass_pred_arr)
    print(f"labels = {labels}")
    # --> labels = [1, 2, 3, 5, 6]

    # the merged binary should be unchanged
    assert clDice_dict["MergedBin"] == {
        "label": "MergedBin",
        "clDice": ((3 / 2) / (7 / 4)),
    }


def test_clDice_Fig51():
    """
    example from Fig 51 of
    Common Limitations of Image Processing Metrics: A Picture Story

    images labeled with label-6, so need to convert to binary

    GT vs Pred 1 clDice = 0.86
    GT vs Pred 2 clDice = 0.67
    """
    # with multiclass label-6
    gt_path = TESTDIR_2D / "shape_6x3_2D_clDice_Fig51_gt.nii.gz"
    pred_1_path = TESTDIR_2D / "shape_6x3_2D_clDice_Fig51_pred_1.nii.gz"
    pred_2_path = TESTDIR_2D / "shape_6x3_2D_clDice_Fig51_pred_2.nii.gz"

    gt_img, gt_arr = load_image_and_array_as_uint8(gt_path)
    pred_1_img, pred_1_arr = load_image_and_array_as_uint8(pred_1_path)
    pred_2_img, pred_2_arr = load_image_and_array_as_uint8(pred_2_path)

    # merged multiclass

    # GT vs Pred 1 ~= 0.86
    assert (
        clDice(
            v_p_pred=convert_multiclass_to_binary(pred_1_arr),
            v_l_gt=convert_multiclass_to_binary(gt_arr),
        )
        == 0.8571428571428571
    )

    # GT vs Pred 2 ~= 0.67
    assert (
        clDice(
            v_p_pred=convert_multiclass_to_binary(pred_2_arr),
            v_l_gt=convert_multiclass_to_binary(gt_arr),
        )
        == 0.6666666666666666
    )

    # log all existing labels
    labels = extract_labels(gt_arr, pred_1_arr)
    print(f"labels = {labels}")
    # --> labels = [6]
    labels = extract_labels(gt_arr, pred_2_arr)
    print(f"labels = {labels}")
    # --> labels = [6]

    # the wrapper clDice_single_label() should be the same as clDice() itself
    # only label-6 present!
    assert clDice_single_label(gt=gt_img, pred=pred_1_img, label=1) == 0
    assert (
        clDice_single_label(gt=gt_img, pred=pred_1_img, label=6) == 0.8571428571428571
    )
    assert clDice_single_label(gt=gt_img, pred=pred_2_img, label=1) == 0
    assert (
        clDice_single_label(gt=gt_img, pred=pred_2_img, label=6) == 0.6666666666666666
    )

    assert clDice_all_classes(track=TRACK.CT, gt=gt_img, pred=pred_1_img) == {
        "6": {"label": "L-ICA", "clDice": 0.8571428571428571},
        "ClsAvgclDice": {"label": "ClsAvgclDice", "clDice": 0.8571428571428571},
        "MergedBin": {"label": "MergedBin", "clDice": 0.8571428571428571},
    }

    assert clDice_all_classes(track=TRACK.MR, gt=gt_img, pred=pred_2_img) == {
        "6": {"label": "L-ICA", "clDice": 0.6666666666666666},
        "ClsAvgclDice": {"label": "ClsAvgclDice", "clDice": 0.6666666666666666},
        "MergedBin": {"label": "MergedBin", "clDice": 0.6666666666666666},
    }


#### multiclass for each of the 2D slice
def test_clDice_all_classes():
    """
    1st slice is shape_6x3_2D_clDice_elong gt vs pred (label-40)
    2nd slice is shape_5x5_2D_clDice_Tshaped gt vs pred (label-39)
    3rd slice is shape_6x3_2D_clDice_Fig51 gt vs pred1 (label-38)
    """
    gt_img = sitk.ReadImage(TESTDIR_3D / "shape_5x6x4_multiclass_clDice_gt.nii.gz")
    pred_img = sitk.ReadImage(TESTDIR_3D / "shape_5x6x4_multiclass_clDice_pred1.nii.gz")

    # CT labels 38-40
    clDice_dict_CT = clDice_all_classes(track=TRACK.CT, gt=gt_img, pred=pred_img)
    assert clDice_dict_CT == {
        "38": {"label": "R-BVR", "clDice": 0.8571428571428571},
        "39": {"label": "L-BVR", "clDice": ((3 / 2) / (7 / 4))},
        "40": {"label": "SSS", "clDice": 1.0},
        "ClsAvgclDice": {"label": "ClsAvgclDice", "clDice": 0.9047619047619048},
        "MergedBin": {"label": "MergedBin", "clDice": 0.8333333333333333},
    }

    # MR labels 38-40
    clDice_dict_MR = clDice_all_classes(track=TRACK.MR, gt=gt_img, pred=pred_img)
    assert clDice_dict_MR == {
        "38": {"label": "L-STA", "clDice": 0.8571428571428571},
        "39": {"label": "R-MaxA", "clDice": ((3 / 2) / (7 / 4))},
        "40": {"label": "L-MaxA", "clDice": 1.0},
        "ClsAvgclDice": {"label": "ClsAvgclDice", "clDice": 0.9047619047619048},
        "MergedBin": {"label": "MergedBin", "clDice": 0.8333333333333333},
    }
