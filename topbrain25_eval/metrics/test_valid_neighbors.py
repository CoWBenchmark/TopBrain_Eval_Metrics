import json
from pathlib import Path

from topbrain25_eval.constants import MUL_CLASS_LABEL_MAP

# Get the current script directory
script_dir = Path(__file__).parent
print(f"script_dir = {script_dir}")


def test_valid_neighbors_ta36_json():
    gt_neighbor_json_filename = "valid_neighbors_ta36.json"
    gt_neighbor_json_path = script_dir / gt_neighbor_json_filename

    # read the gt neighbor dict
    with open(gt_neighbor_json_path) as f:
        gt_neighbors_dict = json.load(f)

    # TA36 keys are 1 to 36
    total_count = len(MUL_CLASS_LABEL_MAP)
    assert list(gt_neighbors_dict.keys()) == [str(x) for x in range(1, total_count)]

    neighbors = []
    for k, v in gt_neighbors_dict.items():
        for node in v:
            neighbors.append((int(k), node))
    print(neighbors)
    # even number of pairs
    assert len(neighbors) % 2 == 0
    # neighbors are symmetric
    for n_pair in neighbors:
        # print(n_pair)
        (n1, n2) = n_pair
        # print((n2, n1))
        assert (n2, n1) in neighbors

    # test left-right symmetry of the neighbors
    pairs = [
        (2, 3),
        (4, 6),  # C6-C7
        (5, 7),  # M1
        (8, 9),
        (11, 12),
        (13, 14),
        (17, 19),  # M2
        (18, 20),  # M3
        (21, 22),
        (23, 24),
        (25, 26),
        (27, 28),
        (29, 30),
        (31, 32),
        (33, 34),
        (35, 36),
    ]
    for a, b in pairs:
        assert len(gt_neighbors_dict[str(a)]) == len(
            gt_neighbors_dict[str(b)]
        ), f"{a, b} not symmetrical!"
