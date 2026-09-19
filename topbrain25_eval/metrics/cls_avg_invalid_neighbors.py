"""
Class-average error on number of invalid neighbors

Each vessel has a list of valid neighbor vessels
Neighborhood is defined by adjacency or "touching"
"""

import json
import os
import tempfile

# import pprint
from pathlib import Path

import SimpleITK as sitk
from topbrain25_eval.for_gc_docker import is_docker
from topbrain25_eval.metrics.generate_cls_avg_dict import generate_cls_avg_dict
from topbrain25_eval.utils.get_neighbor_per_mask import get_neighbor_per_mask

# Use tempfile.NamedTemporaryFile to reserve a file path globally (shared between functions)

# Get the current script directory
if is_docker():
    # in docker, need to read the files outside of module environment
    script_dir = Path("/opt/app/topbrain25_eval/metrics")
else:
    script_dir = Path(__file__).parent
print(f"script_dir = {script_dir}")


def invalid_neighbors_single_label(
    *, gt: sitk.Image, pred: sitk.Image, label: int
) -> int:
    """
    for each label, get its neighbors, and compare
    with its list of valid neighbors
    return the number of invalid neighbors
    """
    # ignore the gt and pred image
    # read from the valid neighbor json and directly diff the number
    # print(f"\n--> invalid_neighbors_single_label() for label-{label}")

    # read the gt neighbor dict
    with open(gt_neighbor_json_path) as f:
        gt_neighbors_dict = json.load(f)
    # read the pred neighbor dict
    with open(pred_neighbor_json_path) as f:
        pred_neighbors_dict = json.load(f)

    # get the list of neighbors for this label
    gt_neighbors = gt_neighbors_dict[str(label)]
    pred_neighbors = pred_neighbors_dict[str(label)]
    # print(f"gt_neighbors = {gt_neighbors}")
    # print(f"pred_neighbors = {pred_neighbors}")

    unique_to_pred = set(pred_neighbors) - set(gt_neighbors)
    # print(f"Elements only in pred: {unique_to_pred}")

    num_invalid_neighbors = len(unique_to_pred)
    # print(f"num_invalid_neighbors = {num_invalid_neighbors}")
    return num_invalid_neighbors


def invalid_neighbors_all_classes(*, gt: sitk.Image, pred: sitk.Image) -> dict:
    """
    use the dict generator from generate_cls_avg_dict
    with invalid_neighbors_single_label() as metric_func
    """
    # run get_neighbor_per_mask() once before single_label() is called
    # then use the saved pred-label-neighbor json each time single_label() is called
    # where we just do the substraction without involving the gt or pred

    # save pred neighbor json path
    global pred_neighbor_json_path

    with tempfile.NamedTemporaryFile(
        prefix="pred_neighbors_", suffix=".json", delete=False, dir=script_dir
    ) as f:
        # multiprocessing-safe, atomically reserve a unique filename
        pred_neighbor_json_path = Path(f.name).resolve()

    print(f"pred_neighbor_json_path = {pred_neighbor_json_path}")

    serializable_dict = get_neighbor_per_mask(pred, pred_neighbor_json_path)

    if not serializable_dict:
        # blank pred

        # clean up
        cleanup_pred_neighbor_json(pred_neighbor_json_path)

        return {
            "ClsAvgNbErr": {"label": "ClsAvgNbErr", "NbErr": 0},
        }

    # read gt valid neighbor json path
    global gt_neighbor_json_path
    gt_neighbor_json_filename = "valid_neighbors_ta36.json"
    gt_neighbor_json_path = (script_dir / gt_neighbor_json_filename).absolute()

    try:
        invalid_neighbors_dict = generate_cls_avg_dict(
            gt=pred,  # NOTE: gt is not used for NbErr
            pred=pred,
            metric_keys=["NbErr"],
            metric_func=invalid_neighbors_single_label,
            binary_merge=False,  # skip binary merged metric
        )
        # print("\ninvalid_neighbors_all_classes() =>")
        # pprint.pprint(invalid_neighbors_dict, sort_dicts=False)
    finally:
        # clean up
        cleanup_pred_neighbor_json(pred_neighbor_json_path)

    return invalid_neighbors_dict


def cleanup_pred_neighbor_json(del_path: Path) -> None:
    print(
        f"[PID {os.getpid()}] deleting {del_path}",
        flush=True,
    )
    del_path.unlink()
