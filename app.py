# Import required libraries
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

import os

from azure.ai.ml.automl import SearchSpace, ClassificationPrimaryMetrics
from azure.ai.ml.sweep import (
    Choice,
    Choice,
    Uniform,
    BanditPolicy,
)

from azure.ai.ml import automl

credential = DefaultAzureCredential()
ml_client = None
try:
    ml_client = MLClient.from_config(credential)
except Exception as ex:
    print(ex)
    # Enter details of your AML workspace
    subscription_id = "2230bd97-57f6-404d-9e75-ecb885c1fe37"
    resource_group = "imag_cup"
    workspace = "abc"
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)


# Uploading image files by creating a 'data asset URI FOLDER':

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml import Input

my_data = Data(
    path="D:\imagine_cup\data",
    type=AssetTypes.URI_FOLDER,
    description="Tomato images",
    name="tomato-images",
)

uri_folder_data_asset = ml_client.data.create_or_update(my_data)

print(uri_folder_data_asset)
print("")
print("Path to folder in Blob Storage:")
print(uri_folder_data_asset.path)

#Convert the downloaded data to JSONL

import json
import os

dataset_parent_dir="D:\imagine_cup"
dataset_dir="D:\imagine_cup\data"

# We'll copy each JSONL file within its related MLTable folder
training_mltable_path = os.path.join(dataset_parent_dir, "training-mltable-folder")
validation_mltable_path = os.path.join(dataset_parent_dir, "validation-mltable-folder")

# First, let's create the folders if they don't exist
os.makedirs(training_mltable_path, exist_ok=True)
os.makedirs(validation_mltable_path, exist_ok=True)

train_validation_ratio = 5

# Path to the training and validation files
train_annotations_file = os.path.join(training_mltable_path, "train_annotations.jsonl")
validation_annotations_file = os.path.join(
    validation_mltable_path, "validation_annotations.jsonl"
)

# Baseline of json line dictionary
json_line_sample = {
    "image_url": uri_folder_data_asset.path,
    "label": "",
}

index = 0
# Scan each sub directary and generate a jsonl line per image, distributed on train and valid JSONL files  # noqa: E501
with open(train_annotations_file, "w") as train_f:
    with open(validation_annotations_file, "w") as validation_f:
        for class_name in os.listdir(dataset_dir):
            sub_dir = os.path.join(dataset_dir, class_name)
            if not os.path.isdir(sub_dir):
                continue

            # Scan each sub directary
            print(f"Parsing {sub_dir}")
            for image in os.listdir(sub_dir):
                json_line = dict(json_line_sample)
                json_line["image_url"] += f"{class_name}/{image}"
                json_line["label"] = class_name

                if index % train_validation_ratio == 0:
                    # validation annotation
                    validation_f.write(json.dumps(json_line) + "\n")
                else:
                    # train annotation
                    train_f.write(json.dumps(json_line) + "\n")
                index += 1

#Create MLTable data input

def create_ml_table_file(filename):
    """Create ML Table definition"""

    return (
        "paths:\n"
        "  - file: ./{0}\n"
        "transformations:\n"
        "  - read_json_lines:\n"
        "        encoding: utf8\n"
        "        invalid_lines: error\n"
        "        include_path_column: false\n"
        "  - convert_column_types:\n"
        "      - columns: image_url\n"
        "        column_type: stream_info"
    ).format(filename)


def save_ml_table_file(output_path, mltable_file_contents):
    with open(os.path.join(output_path, "MLTable"), "w") as f:
        f.write(mltable_file_contents)


# Create and save train mltable
train_mltable_file_contents = create_ml_table_file(
    os.path.basename(train_annotations_file)
)
save_ml_table_file(training_mltable_path, train_mltable_file_contents)

# Create and save validation mltable
validation_mltable_file_contents = create_ml_table_file(
    os.path.basename(validation_annotations_file)
)
save_ml_table_file(validation_mltable_path, validation_mltable_file_contents)

# Training MLTable defined locally, with local data to be uploaded
my_training_data_input = Input(type=AssetTypes.MLTABLE, path=training_mltable_path)

# Validation MLTable defined locally, with local data to be uploaded
my_validation_data_input = Input(type=AssetTypes.MLTABLE, path=validation_mltable_path)

# WITH REMOTE PATH: If available already in the cloud/workspace-blob-store
# my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml://datastores/workspaceblobstore/paths/vision-classification/train")
# my_validation_data_input = Input(type=AssetTypes.MLTABLE, path="azureml://datastores/workspaceblobstore/paths/vision-classification/valid")

#Compute target setup
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError

compute_name = "gpu-cluster-nc6-finalz"

try:
    _ = ml_client.compute.get(compute_name)
    print("Found existing compute target.")
except ResourceNotFoundError:
    print("Creating a new compute target...")
    compute_config = AmlCompute(
        name=compute_name,
        type="amlcompute",
        size="Standard_NC6",
        idle_time_before_scale_down=120,
        min_instances=0,
        max_instances=4,
    )
    ml_client.begin_create_or_update(compute_config).result()
    
#Configure and run the AutoML for Images Classification-MultiClass training job

# set up experiment name
exp_name = "dpv2-tomato-image-classification-experiment"

# Create the AutoML job with the related factory-function.

image_classification_job = automl.image_classification(
    compute=compute_name,
    # name="dpv2-image-classification-job-02",
    experiment_name=exp_name,
    training_data=my_training_data_input,
    validation_data=my_validation_data_input,
    target_column_name="label",
    primary_metric="accuracy",
    tags={"my_custom_tag": "My custom value"},
)

image_classification_job.set_limits(
    max_trials=10,
    max_concurrent_trials=2,
)

#Submitting an AutoML job for Computer Vision tasks
# Submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    image_classification_job
)  # submit the job to the backend

print(f"Created job: {returned_job}")

ml_client.jobs.stream(returned_job.name)


#Manual hyperparameter sweeping for your model

# Create the AutoML job with the related factory-function.

image_classification_job = automl.image_classification(
    compute=compute_name,
    # name="dpv2-image-classification-job-02",
    experiment_name=exp_name,
    training_data=my_training_data_input,
    validation_data=my_validation_data_input,
    target_column_name="label",
    primary_metric=ClassificationPrimaryMetrics.ACCURACY,
    tags={"my_custom_tag": "My custom value"},
)

image_classification_job.set_limits(
    timeout_minutes=60,
    max_trials=10,
    max_concurrent_trials=2,
)

image_classification_job.extend_search_space(
    [
        SearchSpace(
            model_name=Choice(["vitb16r224", "vits16r224"]),
            learning_rate=Uniform(0.001, 0.01),
            number_of_epochs=Choice([15, 30]),
        ),
        SearchSpace(
            model_name=Choice(["seresnext", "resnet50"]),
            learning_rate=Uniform(0.001, 0.01),
            layers_to_freeze=Choice([0, 2]),
        ),
    ]
)

image_classification_job.set_sweep(
    sampling_algorithm="Random",
    early_termination=BanditPolicy(
        evaluation_interval=2, slack_factor=0.2, delay_evaluation=6
    ),
)

# Submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    image_classification_job
)  # submit the job to the backend

print(f"Created job: {returned_job}")

ml_client.jobs.stream(returned_job.name)

hd_job = ml_client.jobs.get(returned_job.name + "_HD")
hd_job

#Retrieve the Best Trial (Best Model's trial/run)

import mlflow  # noqa: E402

# Obtain the tracking URL from MLClient
MLFLOW_TRACKING_URI = ml_client.workspaces.get(
    name=ml_client.workspace_name
).mlflow_tracking_uri

print(MLFLOW_TRACKING_URI)

# Set the MLFLOW TRACKING URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"\nCurrent tracking uri: {mlflow.get_tracking_uri()}")

from mlflow.tracking.client import MlflowClient

# Initialize MLFlow client
mlflow_client = MlflowClient()

#Get the AutoML parent Job
job_name = returned_job.name

# Example if providing an specific Job name/ID
# job_name = "salmon_camel_5sdf05xvb3"

# Get the parent run
mlflow_parent_run = mlflow_client.get_run(job_name)

print("Parent Run: ")
print(mlflow_parent_run)
# Print parent run tags. 'automl_best_child_run_id' tag should be there.
print(mlflow_parent_run.data.tags.keys())

# Get the best model's child run

best_child_run_id = mlflow_parent_run.data.tags["automl_best_child_run_id"]
print(f"Found best child run id: {best_child_run_id}")

best_run = mlflow_client.get_run(best_child_run_id)

print("Best child run: ")
print(best_run)

#Get best model run's metrics

import pandas as pd  # noqa: E402

pd.DataFrame(best_run.data.metrics, index=[0]).T

#Download the best model locally

# Create local folder
import os  # noqa: E402

local_dir = "./artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
# Download run's artifacts/outputs
local_path = mlflow_client.download_artifacts(
    best_run.info.run_id, "outputs", local_dir
)
print(f"Artifacts downloaded in: {local_path}")
print(f"Artifacts: {os.listdir(local_path)}")
import os  # noqa: E402

mlflow_model_dir = os.path.join(local_dir, "outputs", "mlflow-model")

# Show the contents of the MLFlow model folder
os.listdir(mlflow_model_dir)

# You should see a list of files such as the following:
# ['artifacts', 'conda.yaml', 'MLmodel', 'python_env.yaml', 'python_model.pkl', 'requirements.txt']  # noqa: E501

#Register best model and deploy

#Create managed online endpoint

# import required libraries
from azure.ai.ml.entities import (  # noqa: E402
    ManagedOnlineEndpoint,ManagedOnlineDeployment,Model,Environment, # noqa: F401
    CodeConfiguration,  # noqa: F401
    ProbeSettings,
)
# Creating a unique endpoint name with current datetime to avoid conflicts
import datetime  # noqa: E402

online_endpoint_name = "ic-mc-tomato-fruit-items-" + datetime.datetime.now().strftime(
    "%m%d%H%M"
)

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="this is a sample online endpoint for deploying model for tomato fruit",
    auth_mode="key",
    tags={"foo": "bar"},
)
ml_client.begin_create_or_update(endpoint).result()

#Register best model and deploy
model_name = "ic-mc-tomato-fruit-items-model"
model = Model(
    path=f"azureml://jobs/{best_run.info.run_id}/outputs/artifacts/outputs/mlflow-model/",
    name=model_name,
    description="my sample image classification multiclass model for tomato fruit",
    type=AssetTypes.MLFLOW_MODEL,
)

# for downloaded file
# model = Model(
#     path=mlflow_model_dir,
#     name=model_name,
#     description="my sample image classification multiclass model",
#     type=AssetTypes.MLFLOW_MODEL,
# )

registered_model = ml_client.models.create_or_update(model)
registered_model.id


from azure.ai.ml.entities import OnlineRequestSettings  # noqa: E402

req_timeout = OnlineRequestSettings(request_timeout_ms=90000)
deployment = ManagedOnlineDeployment(
    name="ic-mc-tomato-fruit-items-mlflow-deploy",
    endpoint_name=online_endpoint_name,
    model=registered_model.id,
    # use GPU instance type like Standard_NC6s_v3 for faster explanations
    instance_type="Standard_DS3_V2",
    instance_count=1,
    request_settings=req_timeout,
    liveness_probe=ProbeSettings(
        failure_threshold=30,
        success_threshold=1,
        timeout=2,
        period=10,
        initial_delay=2000,
    ),
    readiness_probe=ProbeSettings(
        failure_threshold=10,
        success_threshold=1,
        timeout=10,
        period=10,
        initial_delay=2000,
    ),
)
ml_client.online_deployments.begin_create_or_update(deployment).result()
# ic mc fridge items deployment to take 100% traffic
endpoint.traffic = {"ic-mc-tomato-fruit-items-mlflow-deploy": 100}
ml_client.begin_create_or_update(endpoint).result()

#Get endpoint details
# Get the details for online endpoint
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

# existing traffic details
print(endpoint.traffic)

# Get the scoring URI
print(endpoint.scoring_uri)

# Create request json
import base64

sample_image = os.path.join(dataset_parent_dir, "test", "tom_gr3.jpg")


def read_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


request_json = {
    "input_data": {
        "columns": ["image"],
        "data": [base64.encodebytes(read_image(sample_image)).decode("utf-8")],
    }
}

import json  # noqa: E402

request_file_name = "sample_request_data.json"

with open(request_file_name, "w") as request_file:
    json.dump(request_json, request_file)
resp = ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name=deployment.name,
    request_file=request_file_name,
)

#Visualize detections

import matplotlib  # noqa: E402
import matplotlib_inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import json

IMAGE_SIZE = (18, 12)
plt.figure(figsize=IMAGE_SIZE)
img_np = mpimg.imread(sample_image)
img = Image.fromarray(img_np.astype("uint8"), "RGB")
x, y = img.size

fig, ax = plt.subplots(1, figsize=(15, 15))
# Display the image
ax.imshow(img_np)

prediction = json.loads(resp)[0]
label_index = np.argmax(prediction["probs"])
label = prediction["labels"][label_index]
conf_score = prediction["probs"][label_index]

display_text = f"{label} ({round(conf_score, 3)})"
print(display_text)

color = "red"
plt.text(30, 30, display_text, color=color, fontsize=30)

plt.show()

#Generate the Scores and Explanations

import json

# Define explainability (XAI) parameters
model_explainability = True
xai_parameters = {
    "xai_algorithm": "xrai",
    "visualizations": True,
    "attributions": False,
}

# Create request json
request_json = {
    "input_data": {
        "columns": ["image"],
        "data": [
            json.dumps(
                {
                    "image_base64": base64.encodebytes(read_image(sample_image)).decode(
                        "utf-8"
                    ),
                    "model_explainability": model_explainability,
                    "xai_parameters": xai_parameters,
                }
            )
        ],
    }
}
request_file_name = "sample_request_data.json"

with open(request_file_name, "w") as request_file:
    json.dump(request_json, request_file)
resp = ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name=deployment.name,
    request_file=request_file_name,
)
predictions = json.loads(resp)
predictions

#Visualize Explanations
from io import BytesIO
from PIL import Image


def base64_to_img(base64_img_str):
    base64_img = base64_img_str.encode("utf-8")
    decoded_img = base64.b64decode(base64_img)
    return BytesIO(decoded_img).getvalue()


# visualize explanations of the first image against one of the class
img_bytes = base64_to_img(predictions[0]["visualizations"])
image = Image.open(BytesIO(img_bytes))
