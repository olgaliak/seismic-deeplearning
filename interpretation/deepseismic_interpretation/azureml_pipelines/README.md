# Integrating with AzureML

## Running a Pipeline in AzureML
Set the following environment variables:
```
BLOB_ACCOUNT_NAME
BLOB_CONTAINER_NAME
BLOB_ACCOUNT_KEY
BLOB_SUB_ID
AML_COMPUTE_CLUSTER_NAME
AML_COMPUTE_CLUSTER_MIN_NODES
AML_COMPUTE_CLUSTER_MAX_NODES
AML_COMPUTE_CLUSTER_SKU
```

On Linux:
`export VARIABLE=value`
These can be set automatically in VSCode in an .env file or run with `source .env` in Linux. An example .env file is found at the ROOT of this repo [here](../../../.env.example). You can rename this to .env. Feel free to use this as your .env file but be sure to add this to your .gitignore to ensure you do not commit any secrets. 

Create a .azureml/config.json file in the project's root directory that looks like so:
```json
{
"subscription_id": "<subscription id>",
"resource_group": "<resource group>",
"workspace_name": "<workspace name>"
}

```
At the ROOT of this repo you will find an example [here](../../../.azureml.example/config.json). This is an example please rename the file to .azureml/config.json, input your account information and add this to your .gitignore. 


## Training Pipeline
Here's an example of a possible config file:
```json
{
    "step1":
    {
        "type": "PythonScriptStep",
        "name": "process all files step",
        "script": "process_all_files.py",
        "input_datareference_path": "",
        "input_datareference_name": "raw_input_data",
        "input_dataset_name": "raw_input_data",
        "source_directory": "src/first_preprocess/",
        "arguments": ["--remote_run",
        "--input_path", "input/",
        "--output_path", "normalized_data"],
        "requirements": "src/first_preprocess/preprocess_requirements.txt",
        "node_count": 1,
        "processes_per_node": 1
    },
    "step2":
    {
        "type": "PythonScriptStep",
        "name": "prepare files step",
        "script": "prepare_files.py",
        "input_datareference_path": "normalized_data/",
        "input_datareference_name": "normalized_data_conditioned",
        "input_dataset_name": "normalizeddataconditioned",
        "source_directory": "src/second_preprocess",
        "arguments": ["split_train_val",
        "patch",
        "--label_file", "label.npy",
        "--output_dir", "splits/",
        "--stride=25",
        "--patch=100.",
        "--log_config", "configs/logging.conf"],
        "requirements": "src/second_preprocess/prepare_files_requirements.txt",
        "node_count": 1,
        "processes_per_node": 1,
        "base_image": "pytorch/pytorch"
    },
    "step3":
    {
        "type": "MpiStep",
        "name": "train step",
        "script": "train.py",
        "input_datareference_path": "normalized_data/",
        "input_datareference_name": "normalized_data_conditioned",
        "input_dataset_name": "normalizeddataconditioned",
        "source_directory": "train/",
        "arguments": ["--splits", "splits",
        "--train_data_paths", "normalized_data/file.npy",
        "--label_paths", "label.npy"],
        "requirements": "train/requirements.txt",
        "node_count": 1,
        "processes_per_node": 1,
        "base_image": "pytorch/pytorch"
    }
}
```
  
If you want to create a train pipeline:
1) All of your steps are isolated
    - Your scripts will need to conform to the interface you define in the config file
        - I.e., if step1 is expected to output X and step 2 is expecting X as an input, your scripts need to reflect that
    - If one of your steps has pip package dependencies, make sure it's specified in a requirements.txt file
    - If your script has local dependencies (i.e., is importing from another script) make sure that all dependencies fall underneath the source_directory
2) You have configured your config file to specify the steps needed (see the section below "Configuring a Pipeline" for guidance)

Note: the following arguments are automatically added to any script steps by AzureML:
```--input_data``` and ```--output``` (if output is specified in the pipeline_config.json)
Make sure to add these arguments in your scripts like so:
```python
parser.add_argument('--input_data', type=str, help='path to preprocessed data')
parser.add_argument('--output', type=str, help='output from training')
```
```input_data``` is the absolute path to the input_datareference_path on the blob you specified.
  
# Configuring a Pipeline
  
## Train Pipeline
Define parameters for the run in a config file. See an example [here](pipeline_config.json)
```json
{
    "step1":
    {
        "type": "<type of step. Supported types include PythonScriptStep and MpiStep>",
        "name": "<name in AzureML for this step>",
        "script": "<path to script for this step>",
        "output": "<name of the output in AzureML for this step - optional>",
        "input_datareference_path": "<path on the data reference for the input data - optional>",
        "input_datareference_name": "<name of the data reference in AzureML where the input data lives - optional>",
        "input_dataset_name": "<name of the datastore in AzureML - optional>",
        "source_directory": "<source directory containing the files for this step>",
        "arguments": "<arguments to pass to the script - optional>",
        "requirements": "<path to the requirements.txt file for the step - optional>",
        "node_count": "<number of nodes to run the script on - optional>",
        "processes_per_node": "<number of processes to run on each node - optional>",
        "base_image": "<name of an image registered on dockerhub that you want to use as your base image"
    },
  
    "step2":
    {
        .
        .
        .
    }
}
```
  
## Kicking off a Pipeline
In order to kick off a pipeline, you will need to use the AzureCLI to login to the subscription where your workspace resides:
```bash
az login
az account set -s <subscription id>
```
Kick off the training pipeline defined in your config via your python environment of choice. The code will look like this:
```python
from src.azml.train_pipeline.train_pipeline import TrainPipeline
  
orchestrator = TrainPipeline("<path to your config file>")
orchestrator.construct_pipeline()
run = orchestrator.run_pipeline(experiment_name="DEV-train-pipeline")
```
See an example in [dev/kickoff_train_pipeline.py](dev/kickoff_train_pipeline.py)

If this fails due to access to the Azure ML subscription, you may be able to connect by using a workaround:
Go to [base_pipeline.py](../base_pipeline.py) and add the following import:
```python
from azureml.core.authentication import AzureCliAuthentication
```
Then find the code where we connect to the workspace which looks like this:
```python
self.ws = Workspace.from_config(path=ws_config)
```
and replace it with  this:
```python
cli_auth = AzureCliAuthentication()
self.ws = Workspace(subscription_id=<subscription id>, resource_group=<resource group>, workspace_name=<workspace name>, auth=cli_auth)
```
to get this to run, you will also need to `pip install azure-cli-core`
Then you can go back and follow the instructions above, including az login and setting the subscription, and kick off the pipeline.
  
## Cancelling a Pipeline Run
If you kicked off a pipeline and want to cancel it, run the [cancel_run.py](dev/cancel_run.py) script with the corresponding run_id and step_id. The corresponding run_id and step_id will be printed once you have run the script. You can also find this information when viewing your run in the portal https://portal.azure.com/. If you would prefer to cancel your run in the portal you may also do this as well.