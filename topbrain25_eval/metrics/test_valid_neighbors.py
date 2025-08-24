import json
from pathlib import Path

# Get the current script directory
script_dir = Path(__file__).parent
print(f"script_dir = {script_dir}")


def test_valid_neighbors_ct_all_json():
    gt_neighbor_json_filename = "valid_neighbors_ct_all.json"
    gt_neighbor_json_path = script_dir / gt_neighbor_json_filename

    # read the gt neighbor dict
    with open(gt_neighbor_json_path) as f:
        gt_neighbors_dict = json.load(f)

    # CT keys are 1 to 40
    assert list(gt_neighbors_dict.keys()) == [str(x) for x in range(1, 41)]

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


def test_valid_neighbors_mr_all_json():
    gt_neighbor_json_filename = "valid_neighbors_mr_all.json"
    gt_neighbor_json_path = script_dir / gt_neighbor_json_filename

    # read the gt neighbor dict
    with open(gt_neighbor_json_path) as f:
        gt_neighbors_dict = json.load(f)

    # MR keys are 1 to 42
    assert list(gt_neighbors_dict.keys()) == [str(x) for x in range(1, 43)]

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
