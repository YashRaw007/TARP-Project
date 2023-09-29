# Import required libraries
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

import os
dataset_parent_dir="D:\imagine_cup"
dataset_dir="D:\imagine_cup\data"
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
training_mltable_path = os.path.join(dataset_parent_dir, "training-mltable-folder")
validation_mltable_path = os.path.join(dataset_parent_dir, "validation-mltable-folder")
compute_name = "gpu-cluster-nc6-finalz"
exp_name = "dpv2-tomato-image-classification-experiment"
# Training MLTable defined locally, with local data to be uploaded
my_training_data_input = Input(type=AssetTypes.MLTABLE, path=training_mltable_path)

# Validation MLTable defined locally, with local data to be uploaded
my_validation_data_input = Input(type=AssetTypes.MLTABLE, path=validation_mltable_path)
# We'll copy each JSONL file within its related MLTable folder
training_mltable_path = os.path.join(dataset_parent_dir, "training-mltable-folder")
validation_mltable_path = os.path.join(dataset_parent_dir, "validation-mltable-folder")
#Manual hyperparameter sweeping for your model

#Retrieve the Best Trial (Best Model's trial/run)

import mlflow

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
job_name = "jovial_cushion_vmg0yhbtlg"

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

import pandas as pd

pd.DataFrame(best_run.data.metrics, index=[0]).T

#Download the best model locally

# Create local folder
import os

local_dir = "./artifact_downloads"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)
# Download run's artifacts/outputs
local_path = mlflow_client.download_artifacts(best_run.info.run_id, "outputs", local_dir)
print(f"Artifacts downloaded in: {local_path}")
print(f"Artifacts: {os.listdir(local_path)}")
import os

mlflow_model_dir = os.path.join(local_dir, "outputs", "mlflow-model")

# Show the contents of the MLFlow model folder
os.listdir(mlflow_model_dir)

# You should see a list of files such as the following:
# ['artifacts', 'conda.yaml', 'MLmodel', 'python_env.yaml', 'python_model.pkl', 'requirements.txt']

#Register best model and deploy

#Create managed online endpoint

# import required libraries
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
    ProbeSettings,
)
# Creating a unique endpoint name with current datetime to avoid conflicts
import datetime

online_endpoint_name = "ic-mc-tomato-fruit-items-" + datetime.datetime.now().strftime("%m%d%H%M")

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


from azure.ai.ml.entities import OnlineRequestSettings

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

import json

request_file_name = "sample_request_data.json"

with open(request_file_name, "w") as request_file:
    json.dump(request_json, request_file)
resp = ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name=deployment.name,
    request_file=request_file_name,
)

#Visualize detections

import matplotlib
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
