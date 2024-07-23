import dataclasses

import os
import random
import typing
import logging
import regex as re
import copy

from . import utils
from . import config
from . import runner_host
from . import runner_docker

random.seed(15)
logger = logging.getLogger("api")
__all__ = ["make"]

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
    base_commit  = ["b65489e8f75a1b2a21b6fc1a3c53ddef144e0940"],
    problem_statement = [
        "The tests are failing, because we want to calculate percentage of releases owned per artist and Implement data visualization for artist release percentage. We already proposed some tests, so we need to implement according to them"
    ],
    environment_setup_commit = ["b65489e8f75a1b2a21b6fc1a3c53ddef144e0940"],
    test_patch = ["[]"],
    FAIL_TO_PASS = [
        "['test_calculate_artist_release_percentage (test.fetch_process_collection_test.test_calculate_artist_release_percentage)','test_visualize_music_collection (test.fetch_process_collection_test.test_visualize_artist_release_percentage)','test_update_data_model_and_storage (test.fetch_process_collection_test.test_update_data_model_and_storage)','test_enhance_ui_with_visualization_and_enriched_data (test.fetch_process_collection_test.test_enhance_ui_with_artist_release_percentage_visualization)','test_secure_api_communication (test.fetch_process_collection_test.test_secure_api_communication)','test_optimize_performance_for_fetching_processing_visualization (test.fetch_process_collection_test.test_optimize_performance_for_data_processing_and_visualization)']"
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
        import json

        with open("./dummy_dataset.json", "r") as f:
            return json.load(f)
    if dataset == "apicurl":
        import json

        with open("./apicurl.json", "r") as f:
            return json.load(f)
    else:
        import datasets

        split = None
        if dataset.endswith("/dev") or dataset.endswith("/test"):
            split = dataset.split("/")[-1]
            dataset = "/".join(dataset.split("/")[:-1])
        return datasets.load_dataset(dataset, split=split)


@dataclasses.dataclass
class State:
    repo: typing.Annotated[str, "Repository to be fixed"]
    setup_commit: typing.Annotated[str, "Base commit"]
    path: typing.Annotated[str, "Path to the repository"]
    issue: typing.Annotated[str, "Issue to be fixed"]
    logs: typing.Annotated[typing.Union[str, None], "Logs of previous steps"] = ""
    previous_patches: typing.Annotated[typing.List[str], "Previous patches"] = dataclasses.field(
        default_factory=list
    )
    fail_to_pass: typing.Annotated[typing.List[str], "Tests that currently fails"] = (
        dataclasses.field(default_factory=list)
    )


class InvalidState(State): ...


class Environment:
    def __init__(self, dataset):
        """
        Initialize the environment with a dataset. If the dataset is not available, it will be downloaded lazily.
        """
        self.dataset = dataset
        self.dockerconnector = runner_docker.DockerConnector()
        self.current_index = None
        self.current_path = None
        self.current_issue = None
        self.current_fail_to_pass = None
        self.current_oracle_files = None
        self.current_repo = None
        self.current_commit = None
        self.num_challenges = (  # helper to get the number of issues in the dataset
            self.dataset.num_rows
            if not isinstance(self.dataset, dict)
            else len(self.dataset[list(self.dataset.keys())[0]])
        )

    def reset(self, index: typing.Optional[int] = None) -> State:
        """
        Return a new instance of the selected environment.
        """
        if index is None:
            len_ds = sum(1 for _ in self.dataset["instance_id"])
            index = random.randint(0, len_ds - 1)
        self.current_index = index
        self.current_repo = self.dataset["repo"][self.current_index]
        self.current_issue = self.dataset["problem_statement"][self.current_index]
        self.current_commit = self.dataset["environment_setup_commit"][self.current_index]
        test_patch = self.dataset["test_patch"][self.current_index]

        self.current_path = runner_host.HostEnv.get_environment(
            self.current_repo, self.current_commit
        )
        self.current_fail_to_pass = self._parse_fail_to_pass(
            self.dataset["FAIL_TO_PASS"][self.current_index], self.current_path
        )
        try:
            self.current_oracle_files = self._parse_oracle_text(
                self.dataset["text"][self.current_index]
            )
        except Exception:
            logger.info("No oracle files found", exc_info=True)
            self.current_oracle_files = []
        return State(
            path=self.current_path,
            issue=self.current_issue,
            fail_to_pass=self.current_fail_to_pass,
            previous_patches=[test_patch],
            repo=self.current_repo,
            setup_commit=self.current_commit,
        )

    def step(self, action: typing.Union[str, typing.List[str]], state) -> State:
        """
        Perform an action in the environment.
        """
        if isinstance(action, list):
            return [self.step(a) for a in action]
        if not action:  # Sampler has produced invalid patch
            logger.info("Invalid patch, skipping")
            return InvalidState(**state.__dict__)

        container = self.dockerconnector.get_child_container(
            repo=self.current_repo,
            environment_setup_commit=self.current_commit,
        )
        for patch in state.previous_patches:
            if patch and patch != "[]":
                self.dockerconnector.apply_patch(container, patch=patch)
        self.dockerconnector.apply_patch(container, patch=action)
        log = self.dockerconnector.run_tests(container)
        container.kill()
        new_state = copy.deepcopy(state)
        if new_state.logs:
            new_state.logs.append(log)
        else:
            new_state.logs = [log]

        return new_state

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
        return [runner_host.find_file(root_dir=current_path, filepath=test) for test in tests]
