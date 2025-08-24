import json
import logging
import os
import pprint
from os import PathLike
from pathlib import Path
from typing import Optional

from evalutils import ClassificationEvaluation
from evalutils.exceptions import FileLoaderError
from evalutils.io import SimpleITKLoader
from evalutils.validators import NumberOfCasesValidator
from pandas import DataFrame, concat, merge, set_option

from topbrain25_eval.aggregate.aggregate_all_detection_dicts import (
    aggregate_all_detection_dicts,
)
from topbrain25_eval.constants import TRACK
from topbrain25_eval.for_gc_docker import is_docker, load_predictions_json
from topbrain25_eval.utils.tree_view_dir import DisplayablePath

logger = logging.getLogger(__name__)

# display more content when printing pandas
set_option("display.max_columns", None)
set_option("max_colwidth", None)


class MySegmentationEvaluation(ClassificationEvaluation):
    """
    A special case of classification from evalutils package
    Submission and ground truth are image files (eg, ITK images)
    Same number images in the ground truth dataset as there are in each submission.
    By default, the results per case are also reported.
    """

    def __init__(
        self,
        track: TRACK,
        expected_num_cases: int,
        predictions_path: Optional[PathLike] = None,
        ground_truth_path: Optional[PathLike] = None,
        output_path: Optional[PathLike] = None,
    ):
        self.track = track

        self.execute_in_docker = is_docker()

        print(f"[init] track = {self.track.value}")
        print(f"[init] execute_in_docker = {self.execute_in_docker}")

        if self.execute_in_docker:
            predictions_path = Path("/input/")
            ground_truth_path = Path("/opt/app/ground-truth/")
            output_path = Path("/output/")
            self.predictions_json = Path("/input/predictions.json")
        else:
            # When not in docker environment, the paths of pred, gt etc
            # are set to be on the same level as package dir `topbrain25_eval`
            # Unless they were specified by the user as initialization params

            # Get the path of the current script
            script_path = Path(__file__).resolve()
            print(f"[path] script_path: {script_path}")

            # The resource files (gt, pred etc)
            # are on the same level as package dir `topbrain25_eval`
            # thus are two parents of the current script_path
            resource_path = script_path.parent.parent
            print(f"[path] resource_path: {resource_path}")

            predictions_path = (
                Path(predictions_path)
                if predictions_path is not None
                else resource_path / "predictions/"
            )
            ground_truth_path = (
                Path(ground_truth_path)
                if ground_truth_path is not None
                else resource_path / "ground-truth/"
            )
            output_path = (
                Path(output_path)
                if output_path is not None
                else resource_path / "output/"
            )

        output_file = output_path / "metrics.json"

        # mkdir for out_path if not exist
        Path(output_path).mkdir(parents=True, exist_ok=True)

        print(f"[path] predictions_path={predictions_path}")
        print(f"[path] ground_truth_path={ground_truth_path}")
        print(f"[path] output_file={output_file}")

        # do not proceed if input|predictions|ground-truth folders are empty
        num_input_pred = len(
            [
                str(x)
                for x in predictions_path.rglob("*")
                if x.is_file() and x.name != "predictions.json"
            ]
        )

        num_ground_truth = len(
            [str(x) for x in ground_truth_path.rglob("*") if x.is_file()]
        )

        print(f"num_input_pred = {num_input_pred}")
        print(f"num_ground_truth = {num_ground_truth}")

        assert num_input_pred > 0, "no input files"
        assert num_ground_truth > 0, "no ground truth"

        # early abort if num pred != gt files
        assert num_ground_truth == num_input_pred, "unequal gt & pred"

        for _ in DisplayablePath.make_tree(predictions_path):
            print(_.displayable())
        for _ in DisplayablePath.make_tree(ground_truth_path):
            print(_.displayable())

        # slug for input interface (for gc docker)
        self.slug_input = f"head-{self.track.value}-angiography"

        # set slug for outer interface (for gc docker)
        self.slug_output = f"head-{self.track.value}-angio-segmentation"

        # set file_loader
        # NOTE: SimpleITKLoader is subclass of evalutils ImageLoader
        # ImageLoader returns [{"hash": self.hash_image(img), "path": fname}]
        file_loader = SimpleITKLoader()

        # set self.extensions
        # ground truth rglob for *.nii.gz, prediction rglob for *.mha
        self.extensions = ("*.nii.gz", "*.nii", "*.mha")

        super().__init__(
            ground_truth_path=ground_truth_path,
            predictions_path=predictions_path,
            # use Default: `None` (alphanumerical) for file_sorter_key
            file_sorter_key=None,
            # file_loader is of type FileLoader from evalutils.io
            # segmentation uses SimpleITKLoader
            file_loader=file_loader,
            validators=(
                # we use the NumberOfCasesValidator to check that the correct number
                # of cases has been submitted by the challenge participant
                NumberOfCasesValidator(num_cases=expected_num_cases),
                # NOTE: We do not use UniqueIndicesValidator
                # since this might throw an error due to uuid provided by GC
                # NOTE: also do not use UniqueImagesValidator
            ),
            output_file=output_file,
        )

        print("Path at terminal when executing this file")
        print(os.getcwd() + "\n")

        print("MySegmentationEvaluation __init__ complete!")

    def load(self):
        """
        three input dataframes
        IMPORTANT: we sort them so the rows match!
        then we merge the dataframes so the correct image is loaded
        """
        print("\n-- call load()")
        self._ground_truth_cases = self._load_cases(folder=self._ground_truth_path)
        self._ground_truth_cases = self._ground_truth_cases.sort_values(
            "path"
        ).reset_index(drop=True)

        self._predictions_cases = self._load_cases(folder=self._predictions_path)
        # NOTE: how to sort self._predictions_cases depends on if needs predictions.json
        # using mapping_dict to sort predictions according to ground_truth name
        if self.execute_in_docker:
            # from
            # https://grand-challenge.org/documentation/automated-evaluation/

            # the platform also supplies a JSON file that tells you
            # how to map the algorithm output filenames with the
            # original input interface filenames
            # You as a challenge organizer must, therefore,
            # read /input/predictions.json to map the output filenames
            # with the input filenames
            self.mapping_dict = load_predictions_json(
                fname=self.predictions_json,
                slug_input=self.slug_input,
                slug_output=self.slug_output,
                input_dir=self._predictions_path,
            )
            print("******* self.mapping_dict *******")
            pprint.pprint(self.mapping_dict, sort_dicts=False)
            print("****************************")
            # NOTE: predictions.json used to sort the predictions

            self._predictions_cases["mapping_gt_path"] = [
                self._ground_truth_path / self.mapping_dict[str(path)]
                for path in self._predictions_cases.path
            ]
            self._predictions_cases = self._predictions_cases.sort_values(
                "mapping_gt_path"
            ).reset_index(drop=True)
        else:
            self._predictions_cases = self._predictions_cases.sort_values(
                "path"
            ).reset_index(drop=True)

        print("*** after sorting ***")
        print(f"\nself._ground_truth_cases =\n{self._ground_truth_cases}")

        print(f"\nself._predictions_cases =\n{self._predictions_cases}")

    def _load_cases(self, *, folder: Path) -> DataFrame:
        """
        Overwrite from evalutils.py
        """
        print(f"\n-- call _load_cases(folder={folder})")
        cases = None

        # Use rglob to recursively find all matching extension files,
        # but excluding predictions.json
        files = [
            f
            for ext in self.extensions
            for f in folder.rglob(ext)
            if f.name != "predictions.json"
        ]
        print("rglob files =")
        pprint.pprint(files)

        for f in sorted(files, key=self._file_sorter_key):
            try:
                # class ImageLoader and GenericLoader load() returns
                # [{"hash": self.hash_image(img), "path": fname}]
                new_cases = self._file_loader.load(fname=f)
            except FileLoaderError:
                logger.warning(f"Could not load {f.name} using {self._file_loader}.")
            else:
                if cases is None:
                    cases = new_cases
                else:
                    cases += new_cases

        if cases is None:
            raise FileLoaderError(
                f"Could not load any files in {folder} with {self._file_loader}."
            )

        print("cases = ", cases)
        return DataFrame(cases)

    def validate(self):
        """
        overwrite evalutils
        Validates each dataframe separately
        """
        self._validate_data_frame(df=self._ground_truth_cases)
        self._validate_data_frame(df=self._predictions_cases)

    def merge_ground_truth_and_predictions(self):
        """
        overwrite evalutils merge_ground_truth_and_predictions

        Merge gt, preds files in one df
        """
        print("\n-- call merge_ground_truth_and_predictions()")
        if self._join_key:
            kwargs = {"on": self._join_key}
        else:
            kwargs = {"left_index": True, "right_index": True}

        assert self._ground_truth_cases.shape[0] == self._predictions_cases.shape[0], (
            "different number of cases for gt, pred!"
        )

        # NOTE: indicator=True is crucial
        # otherwise cross_validate(self) will complain!
        self._cases = merge(
            left=self._ground_truth_cases,
            right=self._predictions_cases,
            indicator=True,
            how="outer",
            suffixes=("_ground_truth", "_prediction"),
            **kwargs,
        )

        print("\nmerge_ground_truth_and_predictions =>")
        pprint.pprint(self._cases.to_dict(), sort_dicts=False)

    def score(self):
        """
        Overwrite evalutils score()
        from py3.10 evalutils ClassificationEvaluation()
        """
        print("\n-- call score()")

        # NOTE: the NaN in self._case_results DataFrame comes from concat
        # "Columns outside the intersection will be filled with NaN values"

        # self._case_results is a <class 'pandas.core.frame.DataFrame'>
        self._case_results = DataFrame()
        for idx, case in self._cases.iterrows():
            self._case_results = concat(
                [
                    self._case_results,
                    # self.score_case defined in topbrain25_eval/evaluation.py
                    DataFrame.from_records([self.score_case(idx=idx, case=case)]),
                ],
                ignore_index=True,
            )

        # NOTE: self.score_aggregates() -> dict
        # thus self._aggregate_results is a python dictionary
        self._aggregate_results = self.score_aggregates()

        # # Store the DataFrame as a pickle or temporary file to debug it separately
        # self._case_results.to_pickle("self._case_results.pkl")
        # self._case_results.to_csv("self._case_results.csv")

        # work with self._case_results
        # to post-aggregate detection_dict
        # to get the f1-average
        # add the post-aggregate straight to the
        # self._aggregate_results dict

        # metric-6 Average F1 score
        # detection_dict is under the column `all_detection_dicts`
        dect_avg = aggregate_all_detection_dicts(
            self.track, self._case_results["all_detection_dicts"]
        )

        # add the dection average dict to self._aggregate_results dict
        self._aggregate_results["dect_avg"] = dect_avg

    def save(self):
        """
        Overwrite evalutils save()
        from BaseEvaluation()

        add indentation and sorting

        NOTE: self._metrics is a dict with two sub-dicts:
        def _metrics(self) -> Dict:
            {
                "case": self._case_results.to_dict(),
                "aggregates": self._aggregate_results,
            }
        """
        final_metrics = self._metrics

        if self.execute_in_docker:
            # if in docker environment, hide the "case" field
            # to prevent sensitive info from being leaked
            print("in docker, thus remove case key...")
            del final_metrics["case"]

        pprint.pprint(final_metrics, sort_dicts=False)

        with open(self._output_file, "w") as f:
            f.write(json.dumps(final_metrics, indent=2, sort_keys=True))
