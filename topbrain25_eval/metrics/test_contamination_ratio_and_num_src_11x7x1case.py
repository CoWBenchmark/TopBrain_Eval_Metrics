import pprint
from pathlib import Path

import numpy as np
from contamination_ratio_and_num_src import compute_contamination
from topbrain25_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

TESTDIR = Path("./test_assets/seg_metrics/contamination_ratio_and_n_src/")

FG_metrics = [
    "FGC_ratio",
    "FGC_ratio_after_thresh",
    "FGC_sources",
    "FGC_sources_after_thresh",
    "UnderSeg_ratio",
    "UnderSeg_ratio_after_thresh",
    "UnderSeg_prevalence",
    "UnderSeg_prevalence_after_thresh",
]

BG_metrics = ["BGC_voxels", "BGC_sources"]


def test_contami_11x7x1_im1_bm1_perfect_seg():
    """11x7x1 test case
    interface-margin (im) = 1
    background-margin (bm) = 1
    test for perfect prediction"""

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_11x7x1_GT.nii.gz")
    pred_array = gt_array.copy()

    IM = 1
    BM = 1

    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
        UnderSeg_ratio_thresh=0,
        FGC_ratio_thresh=0,
    )
    pprint.pprint(contami_dict)

    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == contami_dict["FGC_sources"]
        == contami_dict["FGC_sources_after_thresh"]
        == contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == contami_dict["BGC_sources"]
        == contami_dict["BGC_voxels"]
        == 0
    )


def test_contami_11x7x1_im1_bm1_diffThres():
    """11x7x1 test case
    interface-margin (im) = 1
    background-margin (bm) = 1
    test for No threshold vs High threshold"""

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_11x7x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_11x7x1_Pred.nii.gz")

    IM = 1
    BM = 1

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
        UnderSeg_ratio_thresh=0,
        FGC_ratio_thresh=0,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (2 / 9 + 5 / 6) / 3
    )  # 35.2%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 6 / 3
    )  # 2

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (3 / 9) / 3
    )  # 11.1%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 1 / 3
    )  # 33.3%

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 3
    assert contami_dict["BGC_voxels"] == 3

    #####################################################
    #### Higher threshold to filter out noise
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
        UnderSeg_ratio_thresh=0.5,
        FGC_ratio_thresh=0.2,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert contami_dict["FGC_ratio"] == (2 / 9 + 5 / 6) / 3  # 35.2%
    assert contami_dict["FGC_ratio_after_thresh"] == (2 / 9) / 3  # 7.4%

    assert contami_dict["FGC_sources"] == 6 / 3  # 2
    assert contami_dict["FGC_sources_after_thresh"] == 1 / 3  # 0.33

    ######### FG under-segmentation #########
    assert contami_dict["UnderSeg_ratio"] == (3 / 9) / 3  # 11.1%
    assert contami_dict["UnderSeg_ratio_after_thresh"] == 0

    assert contami_dict["UnderSeg_prevalence"] == 1 / 3  # 33.3%
    assert contami_dict["UnderSeg_prevalence_after_thresh"] == 0

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 3
    assert contami_dict["BGC_voxels"] == 3


def test_contami_11x7x1_im1_bm1_withClsFN():
    """11x7x1 test case
    interface-margin (im) = 1
    background-margin (bm) = 1
    one or more FG classes false-negative in pred
    should result in 100% UnderSeg for that class"""

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_11x7x1_GT.nii.gz")
    pred_array = gt_array.copy()

    IM = 1
    BM = 1

    #####################################################
    # Wipe out cls-2 from pred
    pred_array[pred_array == 2] = 0

    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
        UnderSeg_ratio_thresh=0,
        FGC_ratio_thresh=0,
    )
    pprint.pprint(contami_dict)

    # All 0 except class-2 which is 100% under-segmented
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == contami_dict["FGC_sources"]
        == contami_dict["FGC_sources_after_thresh"]
        == contami_dict["BGC_sources"]
        == contami_dict["BGC_voxels"]
        == 0
    )
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 1 / 3
    )

    #####################################################
    # Wipe out both cls-2 and cls-3 from pred
    pred_array[pred_array == 3] = 0

    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
        UnderSeg_ratio_thresh=0,
        FGC_ratio_thresh=0,
    )
    pprint.pprint(contami_dict)

    # All 0 except class-2 AND 3 which are 100% under-segmented
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == contami_dict["FGC_sources"]
        == contami_dict["FGC_sources_after_thresh"]
        == contami_dict["BGC_sources"]
        == contami_dict["BGC_voxels"]
        == 0
    )
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 2 / 3
    )

    #####################################################
    # Wipe out ALL classes from pred
    pred_array[pred_array > 0] = 0
    assert np.all(pred_array == 0)

    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
        UnderSeg_ratio_thresh=0,
        FGC_ratio_thresh=0,
    )
    pprint.pprint(contami_dict)

    # All 0 except UnderSeg_xxx metrics which are 100% under-segmented
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == contami_dict["FGC_sources"]
        == contami_dict["FGC_sources_after_thresh"]
        == contami_dict["BGC_sources"]
        == contami_dict["BGC_voxels"]
        == 0
    )
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 3 / 3  # 100%
    )


def test_contami_11x7x1_im2_bm2_diffThres():
    """11x7x1 test case
    interface-margin (im) = 2
    background-margin (bm) = 2
    test for No threshold vs High threshold"""

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_11x7x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_11x7x1_Pred.nii.gz")

    IM = 2
    BM = 2

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
        UnderSeg_ratio_thresh=0,
        FGC_ratio_thresh=0,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (1 / 4) / 3
    )  # 8.3%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 1 / 3
    )  # 0.33

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (2 / 3) / 3
    )  # 22.2%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 1 / 3
    )  # 33.3%

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 0
    assert contami_dict["BGC_voxels"] == 0

    #####################################################
    #### High but not high enough threshold to filter out noise
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
        UnderSeg_ratio_thresh=0.3,
        FGC_ratio_thresh=0.1,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    # NOTE: FGC_ratio_thresh of 10% is not enough to filter 1/4!
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (1 / 4) / 3
    )  # 8.3%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 1 / 3
    )  # 0.33

    ######### FG under-segmentation #########
    # NOTE: UnderSeg_ratio_thresh of 0.3 is not enought to filter 2/3!
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (2 / 3) / 3
    )  # 22.2%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 1 / 3
    )  # 33.3%

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 0
    assert contami_dict["BGC_voxels"] == 0

    #####################################################
    #### Higher threshold to filter out noise
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
        UnderSeg_ratio_thresh=0.3,
        FGC_ratio_thresh=0.3,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    # NOTE: FGC_ratio_thresh of 30% is enough to filter 1/4
    assert contami_dict["FGC_ratio"] == (1 / 4) / 3  # 8.3%
    assert contami_dict["FGC_ratio_after_thresh"] == 0

    assert contami_dict["FGC_sources"] == 1 / 3  # 0.33
    assert contami_dict["FGC_sources_after_thresh"] == 0

    ######### FG under-segmentation #########
    # NOTE: UnderSeg_ratio_thresh of 0.3 is not enought to filter 2/3!
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (2 / 3) / 3
    )  # 22.2%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 1 / 3
    )  # 33.3%

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 0
    assert contami_dict["BGC_voxels"] == 0


def test_contami_11x7x1_im0_bm0_diffThres():
    """11x7x1 test case
    interface-margin (im) = 0
    background-margin (bm) = 0
    test for No threshold vs High threshold"""

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_11x7x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_11x7x1_Pred.nii.gz")

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        UnderSeg_ratio_thresh=0,
        FGC_ratio_thresh=0,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert contami_dict["FGC_ratio"] == contami_dict["FGC_ratio_after_thresh"]
    assert round(contami_dict["FGC_ratio"], 3) == round(
        (8 / 14 + 6 / 9) / 3, 3
    )  # 41.3%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 6 / 3
    )  # 2

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (3 / 14) / 3
    )  # 7.1%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 1 / 3
    )  # 33.3%

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 3
    assert contami_dict["BGC_voxels"] == 5

    #####################################################
    #### Higher threshold to filter out noise
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        UnderSeg_ratio_thresh=0.2,
        FGC_ratio_thresh=0.2,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert round(contami_dict["FGC_ratio"], 3) == 0.413  # 41.3%
    assert contami_dict["FGC_ratio_after_thresh"] == (7 / 14 + 4 / 9) / 3  # 31.5%

    assert contami_dict["FGC_sources"] == 2
    assert contami_dict["FGC_sources_after_thresh"] == 3 / 3

    ######### FG under-segmentation #########
    # 0.2 not enough to filter 3/14
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (3 / 14) / 3
    )  # 7.1%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 1 / 3
    )  # 33.3%

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 3
    assert contami_dict["BGC_voxels"] == 5
