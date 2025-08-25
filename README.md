# TopBrain 🔝🧠 Evaluation Metrics

This repo contains the package to compute the evaluation metrics for the [**TopBrain 2025 challenge**](https://topbrain2025.grand-challenge.org) on grand-challnge (GC).

## `topbrain25_eval` package

At the root folder, there is a [`pyproject.toml`](./pyproject.toml) config file that can set up the evaluation project folder
as a local pip module called **`topbrain25_eval`** for running the evaluations in your python project.

To setup and install `topbrain25_eval` package:

```sh
# from topbrain25_eval root
bash ./setup.sh

# activate the env with topbrain_eval installed
source env_py310/bin/activate
```

## Run Evaluations with `python3 topbrain25_eval/evaluation.py`

### 1. Configure with `configs.py`

First go to `topbrain25_eval/configs.py` and configure the `track` and `expected_num_cases`.
The `expected_num_cases` is required and must match the number of cases to evalute, i.e. the number of ground-truth cases etc.
See below.

### 2. Folders `ground-truth/` and `predictions/`

When not in docker environment, the paths of pred, gt, roi etc
are set by default to be on the same level as the package dir `topbrain25_eval`:

```sh
# mkdir and put your gt, pred etc like this:
├── ground-truth
├── predictions
├── topbrain25_eval
```

Simply put the files of ground-truth and predictions in the folders `ground-truth/` and `predictions/`,
and run:

```sh
python3 topbrain25_eval/evaluation.py
```

_Note: You can also specify your own custom paths for the ground-truth, predictions, output etc when you call the `TopBrainEvaluation` object in your own code:_

```py
# example from topbrain25_eval/test_evaluation_task_1_seg.py

from topbrain25_eval.evaluation import TopBrainEvaluation

evalRun = TopBrainEvaluation(
    track,
    expected_num_cases,
    predictions_path=predictions_path,
    ground_truth_path=ground_truth_path,
    output_path=output_path,
)

evalRun.evaluate()
```

**The naming of gt and pred files can be _arbitrary as long as their filelist dataframe `.sort_values()` are sorted in the same way_!**

The accepted file formats for ground-truth and predictions are:

- NIfTI (`.nii.gz`, `.nii`) **or** SimpleITK compatible images `.mha` for images and masks

---

### Segmentation metrics

In [`topbrain25_eval/metrics/`](./topbrain25_eval/metrics/), you will find our implementations for evaluating the submitted segmentation predictions.

Six evaluation metrics with equal weights for head-angio multiclass (TopBrain anatomical vessels) segmentation task:

1. Class-average Dice similarity coefficient:
    * [`cls_avg_dice.py`](./topbrain25_eval/metrics/cls_avg_dice.py)
2. Class-average centerline Dice (clDice):
    * [`cls_avg_clDice.py`](./topbrain25_eval/metrics/cls_avg_clDice.py)
3. Class-average error on number of connected components (B0):
    * [`cls_avg_b0.py`](./topbrain25_eval/metrics/cls_avg_b0.py)
4. Class-average Hausdorff Distance 95% Percentile (HD95):
    * [`cls_avg_hd95.py`](./topbrain25_eval/metrics/cls_avg_hd95.py)
5. Class-average error on number of invalid neighbors:
    * [`cls_avg_invalid_neighbors.py`](./topbrain25_eval/metrics/cls_avg_invalid_neighbors.py)
6. Average F1 score (harmonic mean of the precision and recall) for detection of the "side road" vessels:
    * [`detection_sideroad_labels.py`](./topbrain25_eval/metrics/detection_sideroad_labels.py)
    * [`aggregate_all_detection_dicts.py`](./topbrain25_eval/aggregate/aggregate_all_detection_dicts.py)

---

## Unit tests as documentation

The documentations for our code come in the form of unit tests.
Please check our test cases to see the expected inputs and outputs, expected behaviors and calculations.

The files with names that follow the form `test_*.py` contain the test cases for the evaluation metrics.

* Dice:
    * [`test_cls_avg_dice.py`](./topbrain25_eval/metrics/test_cls_avg_dice.py)
* clDice:
    * [`test_cls_avg_clDice.py`](./topbrain25_eval/metrics/test_cls_avg_clDice.py)
* Connected component B0 number error:
    * [`test_cls_avg_b0.py`](./topbrain25_eval/metrics/test_cls_avg_b0.py)
* HD95:
    * [`test_cls_avg_hd95.py`](./topbrain25_eval/metrics/test_cls_avg_hd95.py)
* Invalid neighbor error:
    * [`test_cls_avg_invalid_neighbors.py`](./topbrain25_eval/metrics/test_cls_avg_invalid_neighbors.py)
* detections:
    * [`test_detection_sideroad_labels.py`](./topbrain25_eval/metrics/test_detection_sideroad_labels.py)
    * [`test_aggregate_all_detection_dicts.py`](./topbrain25_eval/aggregate/test_aggregate_all_detection_dicts.py)

Test asset files used in the test cases are stored in the folder [`test_assets/`](./test_assets/).

Simply invoke the tests by `pytest .`:

```bash
# simply run pytest
$ pytest .

topbrain25_eval/aggregate/test_aggregate_all_detection_dicts.py ....                                                                                 [  4%]
topbrain25_eval/metrics/test_cls_avg_b0.py ...............                                                                                           [ 19%]
topbrain25_eval/metrics/test_cls_avg_clDice.py .....                                                                                                 [ 24%]
topbrain25_eval/metrics/test_cls_avg_dice.py .............                                                                                           [ 37%]
topbrain25_eval/metrics/test_cls_avg_hd95.py .............                                                                                           [ 51%]
topbrain25_eval/metrics/test_cls_avg_invalid_neighbors.py ........                                                                                   [ 59%]
topbrain25_eval/metrics/test_detection_sideroad_labels.py ..........                                                                                 [ 69%]
topbrain25_eval/metrics/test_generate_cls_avg_dict.py ..........                                                                                     [ 79%]
topbrain25_eval/metrics/test_valid_neighbors.py ..                                                                                                   [ 81%]
topbrain25_eval/test_constants.py .                                                                                                                  [ 82%]
topbrain25_eval/test_evaluation_task_1_seg.py ..                                                                                                     [ 84%]
topbrain25_eval/test_evaluation_task_1_seg_2.py ..                                                                                                   [ 86%]
topbrain25_eval/test_score_case_task_1_seg.py ..                                                                                                     [ 88%]
topbrain25_eval/utils/test_get_neighbor_per_mask.py .......                                                                                          [ 95%]
topbrain25_eval/utils/test_utils_mask.py ....                                                                                                        [100%]

==================================================================== 98 passed in 3.17s ====================================================================
```
