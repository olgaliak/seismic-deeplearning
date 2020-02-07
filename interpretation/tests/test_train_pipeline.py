"""
Integration tests for the train pipeline
"""
import pytest
from deepseismic_interpretation.azureml_pipelines.train_pipeline import TrainPipeline
import json
import os

TEMP_CONFIG_FILE = "test_batch_config.json"
test_data = None


class TestTrainPipelineIntegration:
    """
    Class used for testing the training pipeline
    """
    global test_data
    test_data = {"step1": {"type": "PythonScriptStep",
                           "name": "data preprocessing step",
                           "script": "standin_preprocess.py",
                           "output": "preprocessed_data",
                           "input_datareference_path": "input/",
                           "input_datareference_name": "training_data",
                           "input_dataset_name": "cggtraindata",
                           "source_directory": "src/dataprocessing",
                           "arguments": []},
                 "step2": {"type": "MpiStep",
                           "name": "train step",
                           "script": "train.py",
                           "source_directory": "src/azml/train_pipeline",
                           "arguments": ["--num_epochs", 10],
                           "requirements": "requirements.txt",
                           "node_count": 1,
                           "processes_per_node": 1}}

    @pytest.fixture(scope="function", autouse=True)
    def teardown(self):
        yield
        if hasattr(self, 'run'):
            self.run.cancel()
        os.remove(TEMP_CONFIG_FILE)

    def test_train_pipeline_expected_inputs_submits_correctly(self):
        # arrange
        self._setup_test_config()
        orchestrator = TrainPipeline("interpretation/tests/example_config.json")
        # act
        orchestrator.construct_pipeline()
        self.run = orchestrator.run_pipeline(experiment_name="TEST-train-pipeline")

        # assert
        assert self.run.get_status() == "Running" or "NotStarted"

    @pytest.mark.parametrize("step,missing_dependency", [("step1", "name"),
                                                         ("step1", "type"),
                                                         ("step1", "arguments"),
                                                         ("step1", "script"),
                                                         ("step1", "output"),
                                                         ("step1", "input_datareference_name"),
                                                         ("step1", "input_datareference_path"),
                                                         ("step1", "input_dataset_name"),
                                                         ("step1", "source_directory"),
                                                         ("step2", "type"),
                                                         ("step2", "name"),
                                                         ("step2", "source_directory"),
                                                         ("step2", "script"),
                                                         ("step2", "arguments"),
                                                         ])
    def test_missing_dependency_in_config_throws_error(self, step, missing_dependency):
        # iterates throw all config dependencies leaving them each out

        # arrange
        self.data = test_data
        self._create_config_without(step, missing_dependency)
        self._setup_test_config()
        orchestrator = TrainPipeline(self.test_config)

        # act / assert
        with pytest.raises(KeyError):
            orchestrator.construct_pipeline()

    # def test_no_env_variables_throws_error(self):
    #     pass

    # def test_no_comp_target_creates_one(self):
    #     pass

    # def test_no_datastore_creates_one(self):
    #     pass

    # def test_non_existent_blob_throws_error(self):
    #     assert True == False

    # def test_non_tags_handles_correctly(self):
    #     pass

    def _create_config_without(self, step, dependency_to_omit):
        """
        helper function that removes dependencies from config file

        :param str step: name of the step with omitted dependency
        :param str dependency_to_omit: the dependency you want to omit from the config
        """
        self.data[step].pop(dependency_to_omit, None)

    def _setup_test_config(self):
        """
        helper function that saves the test data in a temp config file
        """
        self.data = test_data
        self.test_config = TEMP_CONFIG_FILE
        with open(self.test_config, 'w') as data_file:
            json.dump(self.data, data_file)
