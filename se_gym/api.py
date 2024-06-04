from dataclasses import dataclass
import datasets
import os
import random
import typing
import subprocess
import logging

from . import utils
from . import config
from . import runner

random.seed(15)
logger = logging.getLogger("api")
__dict__ = ["make"]

if not os.path.exists(config.DEFAULT_SAVE_PATH):
    os.makedirs(config.DEFAULT_SAVE_PATH)


def make(dataset: str = "princeton-nlp/SWE-bench_Lite_oracle/dev"):
    return Environment(get_ds(dataset))

__dummy_repo = dict(
    repo=["gstenzel/ignore-this-dummy"],
    instance_id=["1"],
    base_commit=["dd707f99dd21d68131c1b97de5c8820f3590cb97"],
    problem_statement=[
        "The magic.main.invert_string function should reverse any string passed to it. But it doesn't. Please fix it."
    ],
    environment_setup_commit=["dd707f99dd21d68131c1b97de5c8820f3590cb97"],
    test_patch=["[]"],
    FAIL_TO_PASS=[
        "['test_string_inversion_1 (test.test_main.test_string_inversion_1)', 'test_string_inversion_2 (test.test_main.test_string_inversion_1)']"
    ],
)

__apicurl = dict(
    repo = ["bodhinsky/apicurl"],
    instance_id =  ["1"],
    base_commit  = ["d281f7f28032db40d42effc18f688bbd698f2af7"],
    problem_statement = [
        config.ISSUE_DESC
    ],
    environment_setup_commit = ["d281f7f28032db40d42effc18f688bbd698f2af7"],
    test_patch = ["[]"],
    FAIL_TO_PASS = [
        "['test_calculate_artist_release_percentage (test.artistOverview_test.test_calculate_artist_release_percentage)','test_visualize_music_collection (test.artistOverview_test.test_visualize_artist_release_percentage)','test_update_data_model_and_storage (test.artistOverview_test.test_update_data_model_and_storage)','test_enhance_ui_with_visualization_and_enriched_data (test.artistOverview_test.test_enhance_ui_with_artist_release_percentage_visualization)','test_secure_api_communication (test.artistOverview_test.test_secure_api_communication)','test_optimize_performance_for_fetching_processing_visualization (test.artistOverview_test.test_optimize_performance_for_data_processing_and_visualization)']"
    ],
)


def get_ds(dataset):
    if dataset == "dummy":
        return __dummy_repo
    if dataset == "apicurl":
        return __apicurl
    else:
        split = None
        if dataset.endswith("/dev") or dataset.endswith("/test"):
            split = dataset.split("/")[-1]
            dataset = "/".join(dataset.split("/")[:-1])
        return datasets.load_dataset(dataset, split=split)


def setup_repo(repo: str, environment_setup_commit: str, test_patch: str = ""):
    logger.debug(f"Setting up repo {repo} at commit {environment_setup_commit}")
    repo_slug = utils.slugify(repo)
    os.makedirs(config.DEFAULT_SAVE_PATH, exist_ok=True)
    target_path = f"{config.DEFAULT_SAVE_PATH}/{repo_slug}"
    if not os.path.exists(f"{config.DEFAULT_SAVE_PATH}/{repo_slug}"):
        subprocess.call(["git", "clone", f"https://github.com/{repo}.git", target_path])
    subprocess.call(config.GIT_DISCARD_CHANGES.split(" "), cwd=target_path)
    subprocess.call(["git", "checkout", environment_setup_commit], cwd=target_path)
    if test_patch:
        logger.debug("Applying test patch")
        with open(f"{target_path}/file.patch", "w") as f:
            f.write(test_patch)
        proc = subprocess.Popen(
            config.GIT_APPLY_PATCH.split(" "),
            cwd=target_path,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        out, err = proc.communicate()
        logger.debug(f"Applied test patch: out {out} err {err}")
        os.unlink(f"{target_path}/file.patch")
        subprocess.call(["git", "add", "."], cwd=target_path)
        subprocess.call(
            ["git", "commit", "-m", "Apply test patch"],
            cwd=target_path,
        )
    return target_path


@dataclass
class State:
    path: typing.Annotated[str, "Path to the repository"]
    issue: typing.Annotated[str, "Issue to be fixed"]
    logs: typing.Annotated[typing.Union[str, None], "Logs of previous steps"] = None
    fail_to_pass: typing.Annotated[typing.List[str], "Tests that currently fails"] = (
        None
    )


class Environment:
    def __init__(self, dataset: datasets.Dataset):
        """
        Initialize the environment with a dataset. If the dataset is not available, it will be downloaded lazily.
        """
        self.dataset = dataset
        self.current_index = None
        self.current_path = None
        self.current_issue = None
        self.test_patch = None
        self.fail_to_pass = None

    def reset(self):
        """
        Return a new instance of the selected environment.
        """
        len_ds = sum(1 for _ in self.dataset["instance_id"])
        self.current_index = random.randint(0, len_ds - 1)
        self.current_path = setup_repo(
            self.dataset["repo"][self.current_index],
            self.dataset["environment_setup_commit"][self.current_index],
            self.dataset["test_patch"][self.current_index],
        )
        self.current_issue = self.dataset["problem_statement"][self.current_index]
        self.test_patch = self.dataset["test_patch"][self.current_index]
        self.fail_to_pass = self.parse_fail_to_pass(
            self.dataset["FAIL_TO_PASS"][self.current_index], self.current_path
        )
        return State(
            path=self.current_path,
            issue=self.current_issue,
            fail_to_pass=self.fail_to_pass,
        )

    def step(self, action: typing.Union[str, typing.List[str]]):
        """
        Perform an action in the environment.
        """
        if isinstance(action, list):
            return [self.step(a) for a in action]
        tree = runner.apply_patch_and_test(
            code_base_root=self.current_path, patch=action
        )
        log = runner.parse_pytest_xml(tree)
        return State(
            path=self.current_path,
            issue=self.current_issue,
            logs=log,
            fail_to_pass=self.fail_to_pass,
        )

    @staticmethod
    def parse_fail_to_pass(fail_to_pass: str, current_path: str) -> typing.List[str]:
        """
        Parse the fail to pass string and return the list of tests that need to be fixed.
        E.g. "['test_boolean_expression_combined (expressions.tests.BasicExpressionsTests)', 'test_boolean_expression_combined_with_empty_Q (expressions.tests.BasicExpressionsTests)']"
        and current_path = "./temp/djangodjango" becomes
        ['temp/djangodjango/tests/expressions/tests.py']
        """
        tests = set()
        for test in eval(fail_to_pass):
            tests.add(
                "/".join(
                    test.split(" ")[-1]
                    .replace("(", "")
                    .replace(")", "")
                    .replace(".", "/")
                    .split("/")[:-1]
                )
                + ".py"
            )
        return [utils.find_file(root_dir=current_path, filename=test) for test in tests]
