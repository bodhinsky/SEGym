from dataclasses import dataclass
import datasets
import os
import random
import typing
import subprocess
import logging
import regex as re

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
    base_commit=["f0ab435cd075b670f311940edf6210dcbb3c745e"],
    problem_statement=[
        "The magic.main.invert_string function should reverse any string passed to it. But it doesn't. Please fix it."
    ],
    environment_setup_commit=["f0ab435cd075b670f311940edf6210dcbb3c745e"],
    test_patch=["[]"],
    text=[
        "Context\n...\n[start of magic/main.py]\n...\n[end of magic/main.py]\n...\n[start of magic/__init__.py]\n...\n[end of magic/__init__.py][start of magic/test/test_main.py]\n...\n[end of magic/test/test_main.py]\n...\nContext\n...\n"
    ],
    FAIL_TO_PASS=[
        "['test_string_inversion_1 (test.test_main.test_string_inversion_1)', 'test_string_inversion_2 (test.test_main.test_string_inversion_1)']"
    ],
)

__apicurl = dict(
    repo = ["bodhinsky/apicurl"],
    instance_id =  ["1"],
    base_commit  = ["bccf189d66bf0ed576ac869926848a0ca5ba9d03"],
    problem_statement = [
        "We want to calculate percentage of releases owned per artist and Implement data visualization for artist release percentage. We already proposed some tests"
    ],
    environment_setup_commit = ["bccf189d66bf0ed576ac869926848a0ca5ba9d03"],
    test_patch = ["[]"],
    FAIL_TO_PASS = [
        "['test_calculate_artist_release_percentage (test.artist_overview_test.test_calculate_artist_release_percentage)','test_visualize_music_collection (test.artist_overview_test.test_visualize_artist_release_percentage)','test_update_data_model_and_storage (test.artist_overview_test.test_update_data_model_and_storage)','test_enhance_ui_with_visualization_and_enriched_data (test.artist_overview_test.test_enhance_ui_with_artist_release_percentage_visualization)','test_secure_api_communication (test.artist_overview_test.test_secure_api_communication)','test_optimize_performance_for_fetching_processing_visualization (test.artist_overview_test.test_optimize_performance_for_data_processing_and_visualization)']"
    ],
)

__swelitemarshmallow1 = dict(
    repo = ["marshmallow-code/marshmallow"],
    instance_id = ["marshmallow-code__marshmallow-1359"],
    base_commit = ["b40a0f4e33823e6d0f341f7e8684e359a99060d1"],
    problem_statement = [
        "3.0: DateTime fields cannot be used as inner field for List or Tuple fields Between releases 3.0.0rc8 and 3.0.0rc9, `DateTime` fields have started throwing an error when being instantiated as inner fields of container fields like `List` or `Tuple`. The snippet below works in <=3.0.0rc8 and throws the error below in >=3.0.0rc9 (and, worryingly, 3.0.0): ```python from marshmallow import fields, Schema class MySchema(Schema): times = fields.List(fields.DateTime()) s = MySchema() ``` Traceback: ``` Traceback (most recent call last): File 'test-mm.py', line 8, in <module> s = MySchema() File '/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py', line 383, in __init__ self.fields = self._init_fields() File '/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py', line 913, in _init_fields self._bind_field(field_name, field_obj) File '/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/schema.py', line 969, in _bind_field field_obj._bind_to_schema(field_name, self) File '/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/fields.py', line 636, in _bind_to_schema self.inner._bind_to_schema(field_name, self) File '/Users/victor/.pyenv/versions/marshmallow/lib/python3.6/site-packages/marshmallow/fields.py', line 1117, in _bind_to_schema or getattr(schema.opts, self.SCHEMA_OPTS_VAR_NAME) AttributeError: 'List' object has no attribute 'opts' ``` It seems like it's treating the parent field as a Schema without checking that it is indeed a schema, so the `schema.opts` statement fails as fields don't have an `opts` attribute."
    ],
    environment_setup_commit = ["b40a0f4e33823e6d0f341f7e8684e359a99060d1"],
    test_patch = ["[]"],
    FAIL_TO_PASS  = [
        "['test_datetime_list_inner_format (tests.test_fields.test_datetime_list_inner_format)']"
    ]
)

__swelitepylint1 = dict(
    repo = ["pylint-dev/astroid"],
    instance_id = ["pylint-dev__astroid-1978"],
    base_commit = ["0c9ab0fe56703fa83c73e514a1020d398d23fa7f"],
    problem_statement = [
        "Deprecation warnings from numpy ### Steps to reproduce 1. Run pylint over the following test case: ``` '''Test case''' import numpy as np value = np.random.seed(1234) ``` ### Current behavior ``` /home/bje/source/nemo/myenv/lib/python3.10/site-packages/astroid/raw_building.py:470: FutureWarning: In the future `np.long` will be defined as the corresponding NumPy scalar. (This may have returned Python scalars in past versions. getattr(sys.modules[modname], name) /home/bje/source/nemo/myenv/lib/python3.10/site-packages/astroid/raw_building.py:470: FutureWarning: In the future `np.long` will be defined as the corresponding NumPy scalar. (This may have returned Python scalars in past versions. getattr(sys.modules[modname], name) ``` ### Expected behavior There should be no future warnings. ### python -c 'from astroid import __pkginfo__; print(__pkginfo__.version)' output 2.12.13"
    ],
    environment_setup_commit = ["0c9ab0fe56703fa83c73e514a1020d398d23fa7f"],
    test_patch = ["[]"],
    FAIL_TO_PASS  = [
        "['test_build_module_getattr_catch_output (tests.unittest_raw_building.test_build_module_getattr_catch_output)']"
    ]
)

def get_ds(dataset):
    if dataset == "dummy":
        return __dummy_repo
    if dataset == "apicurl":
        return __apicurl
    if dataset == "swelitemarshmallow1":
        return __swelitemarshmallow1
    if dataset == "swelitepylint1":
        return __swelitepylint1

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


class InvalidState(State): ...


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
        self.oracle_files = None

    def reset(self) -> State:
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
        self.fail_to_pass = self._parse_fail_to_pass(
            self.dataset["FAIL_TO_PASS"][self.current_index], self.current_path
        )
        try:
            self.oracle_files = self._parse_oracle_text(
                self.dataset["text"][self.current_index]
            )
        except Exception:
            logger.info("No oracle files found", exc_info=True)
            self.oracle_files = []
        return State(
            path=self.current_path,
            issue=self.current_issue,
            fail_to_pass=self.fail_to_pass,
        )

    def step(self, action: typing.Union[str, typing.List[str]]) -> State:
        """
        Perform an action in the environment.
        """
        if isinstance(action, list):
            return [self.step(a) for a in action]
        if not action:  # Sampler has produced invalid patch
            logger.info("Invalid patch, skipping")
            return InvalidState(
                path=self.current_path,
                issue=self.current_issue,
                fail_to_pass=self.fail_to_pass,
            )
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
    def _parse_oracle_text(text: str) -> typing.List[str]:
        pat = re.compile(r"\[start of (.*?)\]")
        return pat.findall(text)

    @staticmethod
    def _parse_fail_to_pass(fail_to_pass: str, current_path: str) -> typing.List[str]:
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
