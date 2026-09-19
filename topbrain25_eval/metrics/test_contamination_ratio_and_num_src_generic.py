import math
import pprint

import numpy as np
from contamination_ratio_and_num_src import compute_contamination
from topbrain25_eval.utils.results_equal import results_equal

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


def test_contami_filled_gt():
    """gt completely filled should result in NaN for BG metrics"""

    gt_array = np.ones((3, 3, 3), dtype=np.uint8)

    contami_dict = compute_contamination(
        gt=gt_array, pred=np.zeros((3, 3, 3), dtype=np.uint8)
    )
    pprint.pprint(contami_dict)

    # FG metrics NOT NaN
    assert all(not math.isnan(contami_dict[k]) for k in FG_metrics)
    # BG metrics all nan
    assert all(math.isnan(contami_dict[k]) for k in BG_metrics)


def test_contami_blank_gt():
    """blank gt should result in NaN for FG metrics"""

    gt_array = np.zeros((3, 3, 3), dtype=np.uint8)

    # regardless of pred_array
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=np.random.choice([0, 1, 2, 3], size=(3, 3, 3)).astype(np.uint8),
    )
    pprint.pprint(contami_dict)

    # FG metrics all NaN
    assert all(math.isnan(contami_dict[k]) for k in FG_metrics)
    # BG metrics NOT nan
    assert all(not math.isnan(contami_dict[k]) for k in BG_metrics)

    # even blank pred_array should still give non-NaN BG metrics (0)
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=np.zeros((3, 3, 3), dtype=np.uint8),
    )
    pprint.pprint(contami_dict)

    # FG metrics all NaN
    assert all(math.isnan(contami_dict[k]) for k in FG_metrics)
    # BG metrics NOT nan
    assert all(contami_dict[k] == 0 for k in BG_metrics)


def test_contami_interface_overshoot_singleVoxTouch():
    """
    When GT fg interface-margin overshoots fg, i.e. when
    GT is thinner than interface-margin, should result in no FG mistakes.

    GT has single voxels touch:
     [[0 0 0]
      [0 2 3]
      [0 0 0]]
    Pred is all 0 blank

    interface-margin = 0 -> underseg
    interface-margin 1, 2, ... -> NO FG mistakes!
    """
    # blank pred
    pred_array = np.zeros((3, 3, 3), dtype=np.uint8)

    # GT cls-2 and cls-3 single voxel touch
    gt_array = np.zeros((3, 3, 3), dtype=np.uint8)
    gt_array[1, 1, 1] = 2
    gt_array[1, 1, 2] = 3

    ###########################################################
    # interface_margin of 0 should result in 100% UnderSeg
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=0,
        fg_fg_interface_method="roll",
    )
    pprint.pprint(contami_dict)
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
        == 1
    )

    ###########################################################
    # interface_margin of 1 or 2 or 3 should result in no FG mistakes
    for im in (1, 2, 3):
        contami_dict = compute_contamination(
            gt=gt_array,
            pred=pred_array,
            fg_fg_interface_margin=im,
            fg_fg_interface_method="roll",
        )
        pprint.pprint(contami_dict)
        # no FG mistakes
        assert all(contami_dict[k] == 0 for k in FG_metrics)
        # also no BG mistakes
        assert all(contami_dict[k] == 0 for k in BG_metrics)


def test_contami_interface_overshoot_singleVoxDoNotTouch():
    """
    GT fg two single voxels do not touch
    [[0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0]
     [0 0 2 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0]
     [0 0 0 0 3 0 0 0 0]
     [0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0]]
    """
    # GT cls-2 and cls-3 single voxel do not touch
    gt_array = np.zeros((3, 8, 9), dtype=np.uint8)
    gt_array[1, 2, 2] = 2
    gt_array[1, 4, 4] = 3

    # pred predicted cls-1 for cls-2 and cls-3
    pred_array = np.zeros((3, 8, 9), dtype=np.uint8)
    pred_array[1, 2, 2] = 1
    pred_array[1, 4, 4] = 1

    ###########################################################
    # interface_margin of 0 or 1 should result in FGC error
    for im in (0, 1):
        contami_dict = compute_contamination(
            gt=gt_array,
            pred=pred_array,
            fg_fg_interface_margin=im,
            fg_fg_interface_method="roll",
        )
        pprint.pprint(contami_dict)
        assert (
            contami_dict["UnderSeg_ratio"]
            == contami_dict["UnderSeg_ratio_after_thresh"]
            == contami_dict["UnderSeg_prevalence"]
            == contami_dict["UnderSeg_prevalence_after_thresh"]
            == contami_dict["BGC_sources"]
            == contami_dict["BGC_voxels"]
            == 0
        )
        assert (
            contami_dict["FGC_ratio"]
            == contami_dict["FGC_ratio_after_thresh"]
            == contami_dict["FGC_sources"]
            == contami_dict["FGC_sources_after_thresh"]
            == 1
        )

    ###########################################################
    # interface_margin of 2 or above should result in no FG mistakes
    for im in (2, 3, 4):
        contami_dict = compute_contamination(
            gt=gt_array,
            pred=pred_array,
            fg_fg_interface_margin=im,
            fg_fg_interface_method="roll",
        )
        pprint.pprint(contami_dict)
        # no FG mistakes
        assert all(contami_dict[k] == 0 for k in FG_metrics)
        # also no BG mistakes
        assert all(contami_dict[k] == 0 for k in BG_metrics)


def test_contami_interface_overshoot_slabsTouch():
    """
    Same as test_contami_interface_overshoot_singleVoxTouch but
    with two slabs of classes touch, and test for much wider margins

    GT cls-2 and cls-3 slabs touch

    Pred is all 0 blank

    interface-margin = 0 -> underseg
    interface-margin 1, 2, ... -> NO FG mistakes!
    """
    # blank pred
    pred_array = np.zeros((8, 9, 10), dtype=np.uint8)

    # GT cls-2 and cls-3 making contact across the x-y plane face
    gt_array = np.zeros((8, 9, 10), dtype=np.uint8)
    gt_array[0, 2:6, 2:7] = 2
    gt_array[1, 2:7, 2:6] = 3

    ###########################################################
    # interface_margin of 0 should result in 100% UnderSeg
    contami_dict = compute_contamination(
        gt=gt_array,
        pred=pred_array,
        fg_fg_interface_margin=0,
        fg_fg_interface_method="roll",
    )
    pprint.pprint(contami_dict)
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
        == 1
    )
    ###########################################################
    # interface_margin of >=1 should result in no FG mistakes
    for im in range(1, 6):
        contami_dict = compute_contamination(
            gt=gt_array,
            pred=pred_array,
            fg_fg_interface_margin=im,
            fg_fg_interface_method="roll",
        )
        pprint.pprint(contami_dict)
        # no FG mistakes
        assert all(contami_dict[k] == 0 for k in FG_metrics)
        # also no BG mistakes
        assert all(contami_dict[k] == 0 for k in BG_metrics)


def test_contami_interface_overshoot_gt_thin():
    # Thinner than the margin along axis 0. Perfect prediction, no bg in GT.
    gt_thin = np.zeros((2, 8, 8), dtype=np.uint8)
    gt_thin[:, :4], gt_thin[:, 4:] = 1, 2
    ###########################################################
    # any interface_margin should result in no FG mistakes and NaN BG metrics
    for im in range(0, 16):
        contami_dict = compute_contamination(
            gt=gt_thin,
            pred=gt_thin.copy(),
            fg_fg_interface_margin=im,
            fg_fg_interface_method="roll",
        )
        pprint.pprint(contami_dict)
        # no FG mistakes
        assert all(contami_dict[k] == 0 for k in FG_metrics)
        # BG metrics all nan
        assert all(math.isnan(contami_dict[k]) for k in BG_metrics)


def roll_eq_filter(gt_array, pred_array):
    contami_dict_roll = compute_contamination(
        gt=gt_array, pred=pred_array, fg_fg_interface_method="roll"
    )
    pprint.pprint(contami_dict_roll)
    contami_dict_filter = compute_contamination(
        gt=gt_array, pred=pred_array, fg_fg_interface_method="filter"
    )
    pprint.pprint(contami_dict_filter)
    # NOTE: Pop or ignore contamination_config before comparing metrics
    contami_dict_roll.pop("contamination_config", None)
    contami_dict_filter.pop("contamination_config", None)
    return results_equal(contami_dict_roll, contami_dict_filter)


def test_contami_fg_fg_interface_methods():
    """The two fg-fg interface methods must give identical results"""
    # filled gt
    gt_array = np.ones((3, 3, 3), dtype=np.uint8)
    pred_array = np.zeros((3, 3, 3), dtype=np.uint8)
    assert roll_eq_filter(gt_array, pred_array)

    # blank gt
    assert roll_eq_filter(pred_array, gt_array)

    # Random GT and Pred
    values = [0, 1, 42, 52]
    shape = (5, 4, 7)
    gt_array = np.random.choice(values, size=shape)
    pred_array = np.random.choice(values, size=shape)
    assert roll_eq_filter(gt_array, pred_array)
