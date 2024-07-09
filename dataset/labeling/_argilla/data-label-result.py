import argilla as rg
from argilla._constants import DEFAULT_API_KEY
from datasets import load_dataset
import json
import os
import sys 
import pandas as pd
import json

if len(sys.argv) < 2:
    print("Usage: python data-label-result.py <config_file_path>")
    sys.exit(1)


api_url = "http://localhost:6900"
api_key = "owner.apikey"
rg.init(api_url=api_url, api_key=api_key)

file_path = sys.argv[1]  # e.g."./dataset/collection/wikipedia/wikipedia.labelconfig.json"
base_path = os.path.dirname(file_path)

config = {}
with open(file_path, "r") as file:
    config = json.load(file)
output_path = config["output"]
remote_dataset_name = os.path.basename(output_path).split(".")[0]
question_names = [question["name"] for question in config["questions"]]

remote_dataset = rg.FeedbackDataset.from_argilla(
    name=remote_dataset_name, workspace="xftest", with_vectors="all"
)

local_dataset = remote_dataset.pull(max_records=1000)
hf_dataset = local_dataset.format_as("datasets")
pandas_dataset = hf_dataset.to_pandas()

def getFirstValue(x):
    if len(x) > 0:
        return x[0]["value"]
    else:
        return None

for question_name in question_names:
    print(question_name)
    pandas_dataset[question_name]=pandas_dataset[question_name].apply(lambda x: getFirstValue(x))

fields = [field.name for field in remote_dataset.fields]
fields.extend(question_names)
pandas_dataset[fields].to_csv(os.path.join(base_path, output_path), index=False)
