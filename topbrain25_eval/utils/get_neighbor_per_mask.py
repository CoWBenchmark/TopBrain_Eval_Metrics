import json
from collections import defaultdict

import numpy as np
import SimpleITK as sitk


def get_neighbor_per_mask(mask_img: sitk.Image, save_json_path) -> dict:
    """
    update the neighbors_dict based on the neighbor from mask_path
    mutate neighbors_dict

    return a serialized dict with sorted list for each label's neighbors
    also save a json file on disk
    """
    shifts = [
        [i, j, k]
        for i in [-1, 0, 1]
        for j in [-1, 0, 1]
        for k in [-1, 0, 1]
        if not (i == 0 and j == 0 and k == 0)
    ]

    # NOTE: SimpleITK npy axis ordering is (z,y,x)!
    # reorder from (z,y,x) to (x,y,z)
    mask_arr = sitk.GetArrayFromImage(mask_img).transpose((2, 1, 0)).astype(np.uint8)

    # pad the original mask image
    label_array = np.pad(mask_arr, pad_width=1)

    # results stored in a dict
    neighbors_dict = defaultdict(set)

    # find the boundaries and only loop over them.
    for shift in shifts:
        shifted_array = np.roll(label_array, shift, axis=(0, 1, 2))

        # Find boundary voxels where the original label is not the same as the shifted label
        # and the original label is not 0 (background)
        boundary_mask = (label_array != shifted_array) & (label_array != 0)

        # Get the labels at the boundary and their neighbors
        boundary_labels = label_array[boundary_mask]
        neighbor_labels = shifted_array[boundary_mask]

        # Add these pairs to the neighbors map
        for lab, neigh_lab in zip(boundary_labels, neighbor_labels):
            lab, neigh_lab = int(lab), int(neigh_lab)
            # print(f"lab, neigh_lab = {lab, neigh_lab}")
            # NOTE: even if no neighbors, value must be an explicit set()
            if lab not in neighbors_dict.keys():
                neighbors_dict[lab] = set()
                # print(f"added {lab}")
            if neigh_lab != 0:  # Exclude background
                neighbors_dict[lab].add(neigh_lab)
                neighbors_dict[neigh_lab].add(lab)  # Neighbors are symmetric

    # type set is not JSON serializable
    # Convert to regular dict with lists
    str_sort_dict = {str(k): sorted(v) for k, v in dict(neighbors_dict).items()}
    serializable_dict = sort_dict_by_numeric_keys(str_sort_dict)

    print("final serializable neighbors_dict =")
    for k, v in serializable_dict.items():
        print(f"{k}: {v}")

    # save the neighbors_dict as a json
    with open(save_json_path, "w") as f:
        f.write(json.dumps((serializable_dict), indent=4))

    print(f"{save_json_path} SAVED!\n")

    return serializable_dict


def sort_dict_by_numeric_keys(data):
    # item here is each tuple, like ('10', [11, 12])
    # Return a new dict sorted by numeric key
    return dict(sorted(data.items(), key=lambda item: int(item[0])))
