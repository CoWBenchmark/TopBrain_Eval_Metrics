import numpy as np
import SimpleITK as sitk
from utils_mask import (
    arr_is_binary,
    convert_multiclass_to_binary,
    extract_labels,
    pad_sitk_image,
)


def test_convert_multiclass_to_binary():
    # np.arange will be converted to all True except first item
    mul_mask = np.arange(6).reshape(3, 2)
    np.testing.assert_equal(
        convert_multiclass_to_binary(mul_mask),
        np.array([[0, 1], [1, 1], [1, 1]]),
    )

    mul_mask = np.arange(27).reshape(3, 3, 3)
    fst_zero = np.ones((3, 3, 3))
    fst_zero[0][0][0] = 0
    np.testing.assert_equal(convert_multiclass_to_binary(mul_mask), fst_zero)

    # all zeroes will stay all zeroes
    np.testing.assert_equal(
        convert_multiclass_to_binary(np.zeros((3, 2, 4))), np.zeros((3, 2, 4))
    )

    # all ones will stay all ones
    np.testing.assert_equal(
        convert_multiclass_to_binary(np.ones((3, 2, 4))), np.ones((3, 2, 4))
    )


def test_extract_labels():
    assert extract_labels(
        array1=np.array([1, 2, 3]),
        array2=np.array([3, 2, 1]),
    ) == [1, 2, 3]

    # will remove background
    assert extract_labels(
        array1=np.array([0, 1, 2, 3]),
    ) == [1, 2, 3]

    # will de-dup the labels
    assert extract_labels(
        array1=np.array([0, 13, 5, 1, 13]),
        array2=np.array([5, 5, 5]),
    ) == [1, 5, 13]

    # wil be sorted
    assert extract_labels(
        array1=np.array([0, 5, 4, 3]),
        array2=np.array([3, 0]),
    ) == [3, 4, 5]


def test_arr_is_binary():
    # all zero or all ones also works!
    assert arr_is_binary(np.zeros((3, 2))) is True
    assert arr_is_binary(np.ones((3, 2))) is True

    # mixed 01
    assert arr_is_binary(np.array([0, 1, 0, 1])) is True

    # bool
    assert arr_is_binary(np.eye(4, dtype=bool)) is True

    # multiclass
    assert arr_is_binary(np.arange(6).reshape(3, 2)) is False
    assert arr_is_binary(np.arange(6).reshape(3, 2).astype(bool)) is True


######
# pad SimpleITK image
def test_pad_sitk_image():
    # a 2D example
    img_2D = sitk.Image([2, 3], sitk.sitkUInt8)
    img_2D = sitk.Add(img_2D, 1)  # Fill with 1s

    padded_2D = pad_sitk_image(img_2D)

    assert np.array_equal(
        sitk.GetArrayFromImage(padded_2D),
        np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        ),
    )
    print("original: ", sitk.GetArrayFromImage(img_2D))

    # a 3D example
    image1 = sitk.Image([2, 2, 2], sitk.sitkUInt8)
    # Fill with 42s
    image1[:, :, :] = 42

    padded = pad_sitk_image(image1)

    assert np.array_equal(
        sitk.GetArrayFromImage(padded),
        np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 42, 42, 0],
                    [0, 42, 42, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 42, 42, 0],
                    [0, 42, 42, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            ]
        ),
    )
    print("original: ", sitk.GetArrayFromImage(image1))
