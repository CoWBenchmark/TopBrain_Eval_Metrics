"""
re-use test_edge cases 003 014 023 in topcow_roi
"""

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from get_neighbor_per_mask import get_neighbor_per_mask, sort_dict_by_numeric_keys
from topbrain25_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

TESTDIR = Path("./test_assets/seg_metrics")


### utility to sort the neighbor list by numeric key
def test_sort_dict_by_numeric_keys():
    input_dict = {
        "11": [4, 10, 12],
        "111": [7, 3],
        "110": [2],
        "10": [9],
        "5": [5, 4],
    }
    # the value list is NOT sorted
    expected_output = {
        "5": [5, 4],
        "10": [9],
        "11": [4, 10, 12],
        "110": [2],
        "111": [7, 3],
    }
    result = sort_dict_by_numeric_keys(input_dict)
    assert list(result.items()) == list(expected_output.items())

    input_dict = {
        "3": [9, 1],
        "1": [3, 2],
        "2": [5, 4],
    }
    # the value list is NOT sorted
    expected_output = {
        "1": [3, 2],
        "2": [5, 4],
        "3": [9, 1],
    }
    result = sort_dict_by_numeric_keys(input_dict)
    assert list(result.items()) == list(expected_output.items())

    input_dict = {
        "1": [2, 3],
        "10": [11, 12],
        "11": [4, 10, 12],
        "12": [6, 10, 11],
        "2": [1, 8],
        "3": [1, 9],
    }
    # sort the key by numeric value 2 -> 10
    expected_output = {
        "1": [2, 3],
        "2": [1, 8],
        "3": [1, 9],
        "10": [11, 12],
        "11": [4, 10, 12],
        "12": [6, 10, 11],
    }
    result = sort_dict_by_numeric_keys(input_dict)
    assert list(result.items()) == list(expected_output.items())


###################################################################
def test_get_neighbor_per_mask_topcow_ct_003():
    test_mask_path = TESTDIR / "topcow_roi/topcow_ct_roi_003.nii.gz"

    mask_img, _ = load_image_and_array_as_uint8(test_mask_path)

    # output dict to rm later
    save_json_path = TESTDIR / "test_get_neighbor_per_mask_topcow_ct_003.json"

    serializable_dict = get_neighbor_per_mask(mask_img, save_json_path)

    assert serializable_dict == {
        "1": [2, 3],
        "2": [1, 8],
        "3": [1, 9],
        "4": [5, 8, 11],
        "5": [4],
        "6": [7, 9],
        "7": [6],
        "8": [2, 4],
        "9": [3, 6],
        "10": [11, 12],
        "11": [4, 10],
        "12": [10],
    }

    # clean up
    os.remove(save_json_path)


def test_get_neighbor_per_mask_topcow_mr_014():
    test_mask_path = TESTDIR / "topcow_roi/topcow_mr_roi_014.nii.gz"

    mask_img, _ = load_image_and_array_as_uint8(test_mask_path)

    # output dict to rm later
    save_json_path = TESTDIR / "test_get_neighbor_per_mask_topcow_mr_014.json"

    serializable_dict = get_neighbor_per_mask(mask_img, save_json_path)

    assert serializable_dict == {
        "1": [2, 3],
        "2": [1],
        "3": [1],
        "4": [11],
        "6": [7, 12],
        "7": [6],
        "10": [11, 12, 15],
        "11": [4, 10, 12],
        "12": [6, 10, 11],
        "15": [10],
    }

    # clean up
    os.remove(save_json_path)


def test_get_neighbor_per_mask_topcow_mr_023():
    test_mask_path = TESTDIR / "topcow_roi/topcow_mr_roi_023.nii.gz"

    mask_img, _ = load_image_and_array_as_uint8(test_mask_path)

    # output dict to rm later
    save_json_path = TESTDIR / "test_get_neighbor_per_mask_topcow_mr_023.json"

    serializable_dict = get_neighbor_per_mask(mask_img, save_json_path)

    assert serializable_dict == {
        "1": [3],
        "2": [8],
        "3": [1],
        "4": [5, 8, 11],
        "5": [4],
        "6": [7],
        "7": [6],
        "8": [2, 4],
        "10": [11, 12],
        "11": [4, 10],
        "12": [10],
    }

    # clean up
    os.remove(save_json_path)


def test_get_label_neighbors_simple_np_mask():
    # Example segmentation mask array
    # 0 - background, 1, 2, 3 - labels
    mask_arr = np.array(
        [
            [0, 1, 1, 0, 2],
            [0, 1, 1, 0, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 3, 3],
        ],
        dtype=np.uint8,
    )
    # convert to 3D
    mask_arr = np.expand_dims(mask_arr, axis=-1)

    mask_img = sitk.GetImageFromArray(mask_arr)

    # output dict to rm later
    save_json_path = TESTDIR / "test_get_label_neighbors_simple_np_mask.json"

    serializable_dict = get_neighbor_per_mask(mask_img, save_json_path)

    assert serializable_dict == {
        # label-1 neighbors = 3
        "1": [3],
        # label-2 neighbors = 3
        "2": [3],
        # label-3 neighbors = 1, 2
        "3": [1, 2],
    }

    # clean up
    os.remove(save_json_path)


def test_get_label_neighbors_complex_np_mask():
    # Example 2D array (a segmentation mask with more labels)
    image_2d = np.array(
        [
            [0, 1, 1, 0, 2, 2, 0],
            [1, 1, 1, 0, 2, 0, 0],
            [4, 4, 0, 0, 0, 0, 3],
            [4, 4, 5, 5, 0, 3, 3],
            [0, 5, 5, 5, 6, 6, 6],
            [7, 7, 0, 6, 6, 0, 7],
        ],
        dtype=np.uint8,
    )
    # convert to 3D
    mask_arr = np.expand_dims(image_2d, axis=-1)

    mask_img = sitk.GetImageFromArray(mask_arr)

    # output dict to rm later
    save_json_path = TESTDIR / "test_get_label_neighbors_complex_np_mask.json"

    serializable_dict = get_neighbor_per_mask(mask_img, save_json_path)

    assert serializable_dict == {
        # label-1 neighbors 4
        "1": [4],
        # label-2 neighbors none
        # 2: set(),
        # label-3 neighbors 6
        "3": [6],
        # label-4 neighbors 1, 5
        "4": [1, 5],
        # label-5 neighbors 4, 6, 7
        "5": [4, 6, 7],
        # label-6 neighbors 3, 5, 7
        "6": [3, 5, 7],
        # label-7 neighbors 5, 6
        "7": [5, 6],
    }

    # clean up
    os.remove(save_json_path)
