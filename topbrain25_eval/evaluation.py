"""
The most important file is evaluation.py.
This is the file where you will extend the Evaluation class
and implement the evaluation for your challenge

inherits BaseEvaluation's .evaluate()
"""

import multiprocessing
import os
import pprint
from concurrent.futures import ProcessPoolExecutor
from os import PathLike
from typing import Optional

from pandas import DataFrame

from topbrain25_eval.aggregate.aggregate_all_detection_dicts import (
    aggregate_all_detection_dicts,
)
from topbrain25_eval.base_algorithm import MySegmentationEvaluation
from topbrain25_eval.score_case_task_1_seg import score_case_task_1_seg
from topbrain25_eval.utils.utils_nii_mha_sitk import access_sitk_attr


class TopBrainEvaluation(MySegmentationEvaluation):
    def __init__(
        self,
        expected_num_cases: int,
        # num_workers 1 as default single-process behavior
        num_workers: int = 1,
        predictions_path: Optional[PathLike] = None,
        ground_truth_path: Optional[PathLike] = None,
        output_path: Optional[PathLike] = None,
    ):
        super().__init__(
            expected_num_cases,
            predictions_path,
            ground_truth_path,
            output_path,
        )
        # cap the max workers
        environ_cpu_limit = os.getenv("GRAND_CHALLENGE_MAX_WORKERS")
        cpu_count = multiprocessing.cpu_count()
        max_allowed = min(int(environ_cpu_limit or cpu_count), cpu_count)
        self.num_workers = min(max_allowed, num_workers)
        print(
            f"start_method = {multiprocessing.get_start_method()}, "
            f"environ_cpu_limit = {environ_cpu_limit}, "
            f"cpu_count = {cpu_count}, "
            f"requested_workers = {num_workers}. ==> "
            f"self.num_workers = {self.num_workers}",
        )

    def score_case(self, *, idx: int, case: DataFrame) -> dict:
        """
        inherits from evalutils BaseEvaluation class

        Loads gt&pred images/files, checks them,
        Send the gt-pred pair to separate
        score_case_task_1.py functions to compute the metrics
        return metrics.json
        """
        pformat_case_to_dict = pprint.pformat(case.to_dict())
        print(
            f"\n[PID {os.getpid()}] -- call score_case(idx={idx})\n"
            f"case =\n{pformat_case_to_dict}\n",
            flush=True,
        )
        gt_path = case["path_ground_truth"]  # from merge() suffixes
        pred_path = case["path_prediction"]  # from merge() suffixes

        # init an empty metrics.json for scocre_case_task* to populate
        metrics_dict = {}

        # Load the images for this case
        # segmentation task uses SimpleITKLoader of ImageLoader
        # which has methods .load_image() and .hash_image()
        gt = self._file_loader.load_image(gt_path)
        pred = self._file_loader.load_image(pred_path)

        # Check that they're the right images
        if (
            self._file_loader.hash_image(gt) != case["hash_ground_truth"]
            or self._file_loader.hash_image(pred) != case["hash_prediction"]
        ):
            raise RuntimeError("Images do not match")

        print("gt original attr:")
        access_sitk_attr(gt)
        print("pred original attr:")
        access_sitk_attr(pred)

        # mutate the metrics_dict by score_case_task_1_seg()
        score_case_task_1_seg(gt=gt, pred=pred, metrics_dict=metrics_dict)

        # add file names
        metrics_dict["pred_fname"] = pred_path.name
        metrics_dict["gt_fname"] = gt_path.name

        return metrics_dict

    def score(self):
        """
        Overrides BaseEvaluation.score() to optionally run score_case()
        across multiple processes.
        """
        if self.num_workers <= 1:
            return super().score()

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit futures in order and keep a mapping to indices
            futures = [
                executor.submit(self.score_case, idx=idx, case=case)
                for idx, case in self._cases.iterrows()
            ]

            # Wait for all to complete
            results = []
            for i, future in enumerate(futures):
                metrics_dict = future.result()
                results.append(metrics_dict)
                print(f"Processed {i + 1}/{len(futures)} cases", flush=True)

        self._case_results = DataFrame(results)

        # the rest from super().score()
        self._aggregate_results = self.score_aggregates()
        dect_avg = aggregate_all_detection_dicts(
            self._case_results["all_detection_dicts"]
        )
        self._aggregate_results["dect_avg"] = dect_avg


if __name__ == "__main__":
    from topbrain25_eval.configs import expected_num_cases, num_workers

    evalRun = TopBrainEvaluation(expected_num_cases, num_workers)

    evalRun.evaluate()

    cowsay_msg = """\n
  ____________________________________
< TopBrainEvaluation().evaluate()  Done! >
  ------------------------------------
         \   ^__^ 
          \  (oo)\_______
             (__)\       )\/\\
                 ||----w |
                 ||     ||
    
    """
    print(cowsay_msg)
