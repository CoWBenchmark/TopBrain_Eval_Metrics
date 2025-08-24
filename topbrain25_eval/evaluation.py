"""
The most important file is evaluation.py.
This is the file where you will extend the Evaluation class
and implement the evaluation for your challenge

inherits BaseEvaluation's .evaluate()
"""

import pprint
from os import PathLike
from typing import Optional

from pandas import DataFrame

from topbrain25_eval.base_algorithm import MySegmentationEvaluation
from topbrain25_eval.constants import TRACK
from topbrain25_eval.score_case_task_1_seg import score_case_task_1_seg
from topbrain25_eval.utils.utils_nii_mha_sitk import access_sitk_attr


class TopBrainEvaluation(MySegmentationEvaluation):
    def __init__(
        self,
        track: TRACK,
        expected_num_cases: int,
        predictions_path: Optional[PathLike] = None,
        ground_truth_path: Optional[PathLike] = None,
        output_path: Optional[PathLike] = None,
    ):
        super().__init__(
            track,
            expected_num_cases,
            predictions_path,
            ground_truth_path,
            output_path,
        )

    def score_case(self, *, idx: int, case: DataFrame) -> dict:
        """
        inherits from evalutils BaseEvaluation class

        Loads gt&pred images/files, checks them,
        Send the gt-pred pair to separate
        score_case_task_1.py functions to compute the metrics
        return metrics.json
        """
        print(f"\n-- call score_case(idx={idx})")
        print("case =\n")
        pprint.pprint(case.to_dict())
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
        score_case_task_1_seg(
            track=self.track, gt=gt, pred=pred, metrics_dict=metrics_dict
        )

        # add file names
        metrics_dict["pred_fname"] = pred_path.name
        metrics_dict["gt_fname"] = gt_path.name

        return metrics_dict


if __name__ == "__main__":
    from topbrain25_eval.configs import expected_num_cases, track

    evalRun = TopBrainEvaluation(track, expected_num_cases)

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
