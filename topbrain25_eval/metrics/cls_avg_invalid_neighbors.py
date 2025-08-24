"""
Class-average error on number of invalid neighbors

Each vessel has a list of valid neighbor vessels
Neighborhood is defined by adjacency or "touching"
"""

import json
import pprint
from pathlib import Path

import SimpleITK as sitk
from topbrain25_eval.constants import TRACK
from topbrain25_eval.metrics.generate_cls_avg_dict import generate_cls_avg_dict
from topbrain25_eval.utils.get_neighbor_per_mask import get_neighbor_per_mask

# # This will store the file path globally (shared between functions)
# _TEMP_PRED_NEIGHBOR_JSON_PATH = None

# Get the current script directory
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
    print(f"\n--> invalid_neighbors_single_label() for label-{label}")

    # read the gt neighbor dict
    with open(gt_neighbor_json_path) as f:
        gt_neighbors_dict = json.load(f)
    # read the pred neighbor dict
    with open(pred_neighbor_json_path) as f:
        pred_neighbors_dict = json.load(f)

    # get the list of neighbors for this label
    gt_neighbors = gt_neighbors_dict[str(label)]
    pred_neighbors = pred_neighbors_dict[str(label)]
    print(f"gt_neighbors = {gt_neighbors}")
    print(f"pred_neighbors = {pred_neighbors}")

    unique_to_pred = set(pred_neighbors) - set(gt_neighbors)
    print(f"Elements only in pred: {unique_to_pred}")

    num_invalid_neighbors = len(unique_to_pred)
    print(f"num_invalid_neighbors = {num_invalid_neighbors}")
    return num_invalid_neighbors


def invalid_neighbors_all_classes(
    *, track: TRACK, gt: sitk.Image, pred: sitk.Image
) -> dict:
    """
    use the dict generator from generate_cls_avg_dict
    with invalid_neighbors_single_label() as metric_func
    """
    # run get_neighbor_per_mask() once before single_label() is called
    # then use the saved pred-label-neighbor json each time single_label() is called
    # where we just do the substraction without involving the gt or pred

    # with tempfile.NamedTemporaryFile() as tmp_file:
    #     tmp_file.write(b"Temporary hello")

    # save pred neighbor json path
    global pred_neighbor_json_path
    pred_neighbor_json_path = (script_dir / "pred_neighbors.json").absolute()

    serializable_dict = get_neighbor_per_mask(pred, pred_neighbor_json_path)

    # read gt valid neighbor json path
    global gt_neighbor_json_path
    gt_neighbor_json_filename = f"valid_neighbors_{track.value}_all.json"
    gt_neighbor_json_path = (script_dir / gt_neighbor_json_filename).absolute()

    invalid_neighbors_dict = generate_cls_avg_dict(
        track=track,
        gt=pred,  # NOTE: gt is not used for NbErr
        pred=pred,
        metric_keys=["NbErr"],
        metric_func=invalid_neighbors_single_label,
        binary_merge=False,  # skip binary merged metric
    )
    print("\ninvalid_neighbors_all_classes() =>")
    pprint.pprint(invalid_neighbors_dict, sort_dicts=False)
    return invalid_neighbors_dict
