import pprint
from pathlib import Path

from contamination_ratio_and_num_src import compute_contamination
from topbrain25_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

TESTDIR = Path("./test_assets/seg_metrics/contamination_ratio_and_n_src/")


# NOTE: Pred A,B,C when im=bm=2, produce the same contamination metrics


def test_contami_8x4x1_im0_bm0_PreA():
    """8x4x1 test case Pred A
    interface-margin (im) = 0
    background-margin (bm) = 0

    No relaxation: class 1 has 8 voxels, class 2 has 10 voxels
    """

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_Pred_A.nii.gz")

    IM = 0
    BM = 0

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (2 / 8 + 4 / 10) / 2
    )  # 32.5%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 3 / 2
    )  # 1.5

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (2 / 8 + 1 / 10) / 2
    )  # 17.5%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 2 / 2
    )  # 1

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 2
    assert contami_dict["BGC_voxels"] == 5


def test_contami_8x4x1_im0_bm0_PreB():
    """8x4x1 test case Pred B
    interface-margin (im) = 0
    background-margin (bm) = 0
    """

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_Pred_B.nii.gz")

    IM = 0
    BM = 0

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (2 / 8 + 4 / 10) / 2
    )  # 32.5%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 3 / 2
    )  # 1.5

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (6 / 8 + 1 / 10) / 2
    )  # 42.5%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 2 / 2
    )  # 1

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 1
    assert contami_dict["BGC_voxels"] == 3


def test_contami_8x4x1_im0_bm0_PreC():
    """8x4x1 test case Pred C
    interface-margin (im) = 0
    background-margin (bm) = 0
    """

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_Pred_C.nii.gz")

    IM = 0
    BM = 0

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (2 / 8 + 3 / 10) / 2
    )  # 27.5%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 3 / 2
    )  # 1.5

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (2 / 8 + 1 / 10) / 2
    )  # 17.5%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 2 / 2
    )  # 1

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 2
    assert contami_dict["BGC_voxels"] == 3


def test_contami_8x4x1_im1_bm1_PreA():
    """8x4x1 test case Pred A
    interface-margin (im) = 1
    background-margin (bm) = 1

    7 fg voxels are on the 1-2 interface: class 1 keeps 5 voxels, class 2 keeps 6 voxels.
    Only 4 bg voxels (the 2x1 blocks at the two upper corners) are not on the bg surface.
    """

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_Pred_A.nii.gz")

    IM = 1
    BM = 1

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (1 / 5 + 3 / 6) / 2
    )  # 35%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 3 / 2
    )  # 1.5

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (2 / 5 + 1 / 6) / 2
    )  # 28.3%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 2 / 2
    )  # 1

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 2
    assert contami_dict["BGC_voxels"] == 3


def test_contami_8x4x1_im1_bm1_PreB():
    """8x4x1 test case Pred B
    interface-margin (im) = 1
    background-margin (bm) = 1
    """

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_Pred_B.nii.gz")

    IM = 1
    BM = 1

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (1 / 5 + 3 / 6) / 2
    )  # 35%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 3 / 2
    )  # 1.5

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (4 / 5 + 1 / 6) / 2
    )  # 48.3%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 2 / 2
    )  # 1

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 1
    assert contami_dict["BGC_voxels"] == 2


def test_contami_8x4x1_im1_bm1_PreC():
    """8x4x1 test case Pred C
    interface-margin (im) = 1
    background-margin (bm) = 1
    """

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_Pred_C.nii.gz")

    IM = 1
    BM = 1

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (1 / 5 + 2 / 6) / 2
    )  # 26.7%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 3 / 2
    )  # 1.5

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (2 / 5 + 1 / 6) / 2
    )  # 28.3%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 2 / 2
    )  # 1

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 2
    assert contami_dict["BGC_voxels"] == 2


def test_contami_8x4x1_im2_bm2_PreA():
    """8x4x1 test case Pred A
    interface-margin (im) = 2
    background-margin (bm) = 2
    """

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_Pred_A.nii.gz")

    IM = 2
    BM = 2

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (1 / 2) / 2
    )  # 25%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 1 / 2
    )  # 0.5

    # ensure there are two voxels after margin=2 relaxation
    assert contami_dict["fg_con_ratios_dict_debug"] == {
        1: {0: {"n_error_voxels": 2, "n_gt_voxels": 2, "ratio": 1.0}},
        2: {
            0: {"n_error_voxels": 1, "n_gt_voxels": 2, "ratio": 0.5},
            3: {"n_error_voxels": 1, "n_gt_voxels": 2, "ratio": 0.5},
        },
    }

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (2 / 2 + 1 / 2) / 2
    )  # 75%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 2 / 2
    )  # 1

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 0
    assert contami_dict["BGC_voxels"] == 0


def test_contami_8x4x1_im2_bm2_PreB():
    """8x4x1 test case Pred B
    interface-margin (im) = 2
    background-margin (bm) = 2
    """

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_Pred_B.nii.gz")

    IM = 2
    BM = 2

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (1 / 2) / 2
    )  # 25%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 1 / 2
    )  # 0.5

    # ensure there are two voxels after margin=2 relaxation
    assert contami_dict["fg_con_ratios_dict_debug"] == {
        1: {0: {"n_error_voxels": 2, "n_gt_voxels": 2, "ratio": 1.0}},
        2: {
            0: {"n_error_voxels": 1, "n_gt_voxels": 2, "ratio": 0.5},
            3: {"n_error_voxels": 1, "n_gt_voxels": 2, "ratio": 0.5},
        },
    }

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (2 / 2 + 1 / 2) / 2
    )  # 75%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 2 / 2
    )  # 1

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 0
    assert contami_dict["BGC_voxels"] == 0


def test_contami_8x4x1_im2_bm2_PreC():
    """8x4x1 test case Pred C
    interface-margin (im) = 2
    background-margin (bm) = 2
    """

    _, gt_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_GT.nii.gz")
    _, pred_array = load_image_and_array_as_uint8(TESTDIR / "lps_8x4x1_Pred_C.nii.gz")

    IM = 2
    BM = 2

    #####################################################
    #### No threshold for UnderSeg and FGC_ratio
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=IM,
        bg_surface_margin=BM,
    )
    pprint.pprint(contami_dict)

    ######### FG contamination #########
    assert (
        contami_dict["FGC_ratio"]
        == contami_dict["FGC_ratio_after_thresh"]
        == (1 / 2) / 2
    )  # 25%
    assert (
        contami_dict["FGC_sources"] == contami_dict["FGC_sources_after_thresh"] == 1 / 2
    )  # 0.5

    # ensure there are two voxels after margin=2 relaxation
    assert contami_dict["fg_con_ratios_dict_debug"] == {
        1: {0: {"n_error_voxels": 2, "n_gt_voxels": 2, "ratio": 1.0}},
        2: {
            0: {"n_error_voxels": 1, "n_gt_voxels": 2, "ratio": 0.5},
            3: {"n_error_voxels": 1, "n_gt_voxels": 2, "ratio": 0.5},
        },
    }

    ######### FG under-segmentation #########
    assert (
        contami_dict["UnderSeg_ratio"]
        == contami_dict["UnderSeg_ratio_after_thresh"]
        == (2 / 2 + 1 / 2) / 2
    )  # 75%
    assert (
        contami_dict["UnderSeg_prevalence"]
        == contami_dict["UnderSeg_prevalence_after_thresh"]
        == 2 / 2
    )  # 1

    ######### BG contamination #########
    assert contami_dict["BGC_sources"] == 0
    assert contami_dict["BGC_voxels"] == 0
