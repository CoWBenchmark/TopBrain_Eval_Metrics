import SimpleITK as sitk

from topbrain25_eval.constants import TRACK
from topbrain25_eval.metrics.cls_avg_b0 import betti_number_error_all_classes
from topbrain25_eval.metrics.cls_avg_clDice import clDice_all_classes
from topbrain25_eval.metrics.cls_avg_dice import dice_coefficient_all_classes
from topbrain25_eval.metrics.cls_avg_hd95 import hd95_all_classes
from topbrain25_eval.metrics.cls_avg_invalid_neighbors import (
    invalid_neighbors_all_classes,
)
from topbrain25_eval.metrics.detection_sideroad_labels import detection_sideroad_labels
from topbrain25_eval.metrics.generate_cls_avg_dict import update_metrics_dict


def score_case_task_1_seg(
    track: TRACK, gt: sitk.Image, pred: sitk.Image, metrics_dict: dict
) -> None:
    """
    score_case() for Task-1-Multiclass-Head-Angio-Segmentation

    work with image gt pred and mutate the metrics_dict object
    """
    # Cast to the same type
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkUInt8)
    caster.SetNumberOfThreads(1)
    gt = caster.Execute(gt)
    pred = caster.Execute(pred)

    # make sure they have the same metadata
    # Copies the Origin, Spacing, and Direction from the gt image
    # NOTE: metadata like image.GetPixelIDValue() and image.GetPixelIDTypeAsString()
    # are NOT copied from source image
    pred.CopyInformation(gt)

    # Score the case

    # (1) add Dice for each class
    dice_dict = dice_coefficient_all_classes(track=track, gt=gt, pred=pred)
    for key in dice_dict:
        update_metrics_dict(
            cls_avg_dict=dice_dict,
            metrics_dict=metrics_dict,
            key=key,
            metric_name="Dice",
        )

    # (2) add clDice for each class
    clDice_dict = clDice_all_classes(track=track, gt=gt, pred=pred)
    for key in clDice_dict:
        update_metrics_dict(
            cls_avg_dict=clDice_dict,
            metrics_dict=metrics_dict,
            key=key,
            metric_name="clDice",
        )

    # (3) add Betti0 number error for each class
    betti_num_err_dict = betti_number_error_all_classes(track=track, gt=gt, pred=pred)
    for key in betti_num_err_dict:
        # no more Betti_1 for topcow24
        # and we are interested in the class-average B0 error per case
        update_metrics_dict(
            cls_avg_dict=betti_num_err_dict,
            metrics_dict=metrics_dict,
            key=key,
            metric_name="B0err",
        )

    # (4) add HD95 and HD for each class
    hd_dict = hd95_all_classes(track=track, gt=gt, pred=pred)
    for key in hd_dict:
        # class-average key is singular for HD or HD95
        if key == "ClsAvgHD":
            # HD
            update_metrics_dict(
                cls_avg_dict=hd_dict,
                metrics_dict=metrics_dict,
                key=key,
                metric_name="HD",
            )
        elif key == "ClsAvgHD95":
            # HD95
            update_metrics_dict(
                cls_avg_dict=hd_dict,
                metrics_dict=metrics_dict,
                key=key,
                metric_name="HD95",
            )
        else:
            # both HD and HD95 for individual CoW labels
            # HD
            update_metrics_dict(
                cls_avg_dict=hd_dict,
                metrics_dict=metrics_dict,
                key=key,
                metric_name="HD",
            )
            # HD95
            update_metrics_dict(
                cls_avg_dict=hd_dict,
                metrics_dict=metrics_dict,
                key=key,
                metric_name="HD95",
            )

    # (5) add invalid neighbor error for each class
    invalid_neighbors_dict = invalid_neighbors_all_classes(
        track=track, gt=gt, pred=pred
    )
    for key in invalid_neighbors_dict:
        update_metrics_dict(
            cls_avg_dict=invalid_neighbors_dict,
            metrics_dict=metrics_dict,
            key=key,
            metric_name="NbErr",
        )

    # (6) add side-road vessel detections
    # each case will have its own detection_dict
    # altogether will be a column of detection_dicts
    # thus name the column as `all_detection_dicts`
    detection_dict = detection_sideroad_labels(track=track, gt=gt, pred=pred)
    metrics_dict["all_detection_dicts"] = detection_dict
