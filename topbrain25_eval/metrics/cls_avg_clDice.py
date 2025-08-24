"""
Class-average centerline Dice (clDice)
"""

import pprint

import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize, skeletonize_3d
from topbrain25_eval.constants import TRACK
from topbrain25_eval.metrics.generate_cls_avg_dict import generate_cls_avg_dict
from topbrain25_eval.utils.utils_mask import convert_multiclass_to_binary


def cl_score(*, s_skeleton: np.array, v_image: np.array) -> float:
    """[this function computes the skeleton volume overlap]
    Args:
        s ([bool]): [skeleton]
        v ([bool]): [image]
    Returns:
        [float]: [computed skeleton volume intersection]

    meanings of v, s refer to clDice paper:
    https://arxiv.org/abs/2003.07311
    """
    if np.sum(s_skeleton) == 0:
        return 0
    return float(np.sum(s_skeleton * v_image) / np.sum(s_skeleton))


def clDice(*, v_p_pred: np.array, v_l_gt: np.array) -> float:
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]

    meanings of v_l, v_p, s_l, s_p refer to clDice paper:
    https://arxiv.org/abs/2003.07311
    """
    # NOTE: skeletonization works on binary images;
    # need to convert multiclass to binary mask first
    pred_mask = convert_multiclass_to_binary(v_p_pred)
    gt_mask = convert_multiclass_to_binary(v_l_gt)

    # clDice makes use of the skimage skeletonize method
    # see https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#skeletonize
    if len(pred_mask.shape) == 2:
        call_skeletonize = skeletonize
    elif len(pred_mask.shape) == 3:
        call_skeletonize = skeletonize_3d

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=call_skeletonize(pred_mask), v_image=gt_mask)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=call_skeletonize(gt_mask), v_image=pred_mask)

    if (tprec + tsens) == 0:
        return 0

    return 2 * tprec * tsens / (tprec + tsens)


def clDice_single_label(*, gt: sitk.Image, pred: sitk.Image, label: int) -> float:
    """
    wrapper for clDice() similar to b0/dice/hd95_single_label()
    """
    print(f"\nfor label-{label}")

    # NOTE: SimpleITK npy axis ordering is (z,y,x)!
    # reorder from (z,y,x) to (x,y,z)
    # This is required for clDice numpy array input
    gt_label_arr = (
        sitk.GetArrayFromImage(gt == label).transpose((2, 1, 0)).astype(np.uint8)
    )
    pred_label_arr = (
        sitk.GetArrayFromImage(pred == label).transpose((2, 1, 0)).astype(np.uint8)
    )

    # Check if label exists for both gt and pred
    # If not, clDice is automatically set to 0 due to FP or FN
    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        print(f"[!!Warning] label-{label} empty for gt or pred")
        return 0
    else:
        return clDice(v_p_pred=pred_label_arr, v_l_gt=gt_label_arr)


def clDice_all_classes(*, track: TRACK, gt: sitk.Image, pred: sitk.Image) -> dict:
    """
    use the dict generator from generate_cls_avg_dict
    with clDice_single_label() as metric_func
    """
    clDice_dict = generate_cls_avg_dict(
        track=track,
        gt=gt,
        pred=pred,
        metric_keys=["clDice"],
        metric_func=clDice_single_label,
    )
    print("\nclDice_all_classes() =>")
    pprint.pprint(clDice_dict, sort_dicts=False)
    return clDice_dict
