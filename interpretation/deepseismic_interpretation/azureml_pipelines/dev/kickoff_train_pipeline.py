"""
Create pipeline and kickoff run
"""
from src.azml.train_pipeline.train_pipeline import TrainPipeline

orchestrator = TrainPipeline("train/pipeline_config.json")
orchestrator.construct_pipeline()
run = orchestrator.run_pipeline(experiment_name="DEV-train-pipeline")
