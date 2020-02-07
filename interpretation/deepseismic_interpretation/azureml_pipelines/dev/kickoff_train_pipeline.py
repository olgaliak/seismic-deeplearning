"""
Create pipeline and kickoff run
"""
from deepseismic_interpretation.azureml_pipelines.train_pipeline import TrainPipeline

orchestrator = TrainPipeline("interpretation/deepseismic_interpretation/azureml_pipelines/pipeline_config.json")
orchestrator.construct_pipeline()
run = orchestrator.run_pipeline(experiment_name="DEV-train-pipeline")
