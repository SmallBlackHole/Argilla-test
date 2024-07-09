import argilla as rg
from argilla._constants import DEFAULT_API_KEY
from datasets import load_dataset
import json
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python data-label-result.py <config_file_path>")
    sys.exit(1)

api_url = "https://aiacalephr.lemonfield-112d55e0.westus3.azurecontainerapps.io/"
api_key = "owner.apikey"  # admin.apikey
rg.init(api_url=api_url, api_key=api_key)

# workspace = rg.Workspace.create("xftest")

file_path = sys.argv[
    1
]  # e.g."./dataset/collection/wikipedia/wikipedia.datalabeling.config.json"
base_path = os.path.dirname(file_path)

config = {}
with open(file_path, "r") as file:
    config = json.load(file)

dataset_file_path = config["dataset"]

hf_dataset = load_dataset(
    "csv", data_files=os.path.join(base_path, dataset_file_path), split="train"
)
records = []

for record in hf_dataset:
    fields = {}
    for column in hf_dataset.column_names:
        fields[column.lower()] = record[column]

    records.append(
        rg.FeedbackRecord(
            fields=fields,
            metadata={},
        )
    )

users = [u for u in rg.User.list()]

from argilla.client.feedback.utils import assign_records

assignments = assign_records(users=users, records=records, overlap=1, shuffle=True)

# Add the metadata to the existing records using id to identify each record
id_modified_records = {}
for username, records in assignments.items():
    for record in records:
        record_id = id(record)
        if record_id not in id_modified_records:
            id_modified_records[record_id] = record
            record.metadata["annotators"] = []
        if username not in id_modified_records[record_id].metadata["annotators"]:
            id_modified_records[record_id].metadata["annotators"].append(username)

# Get the unique records with their updated metadata
modified_records = list(id_modified_records.values())


# Push the dataset with the modified records
dataset = rg.FeedbackDataset(
    fields=[rg.TextField(name=column.lower()) for column in hf_dataset.column_names],
    questions=[
        (
            rg.RatingQuestion(
                name=question["name"],
                description=question["description"],
                values=question["values"],
            )
            if question["type"] == "RatingQuestion"
            else (
                rg.TextQuestion(
                    name=question["name"],
                    description=question["description"],
                )
                if question["type"] == "TextQuestion"
                else rg.LabelQuestion(
                    name=question["name"],
                    description=question["description"],
                    labels=question["labels"],
                )
            )
        )
        for question in config["questions"]
        if question["type"] in ["RatingQuestion", "TextQuestion", "LabelQuestion"]
    ],
    metadata_properties=[rg.TermsMetadataProperty(name="annotators")],
    guidelines="Please, read the question carefully and try to answer it as accurately as possible.",
)
dataset.add_records(modified_records)
dataset_name = os.path.basename(config["output"]).split(".")[0]
remote_dataset = dataset.push_to_argilla(name=dataset_name, workspace="xftest")
config["id"] = str(remote_dataset.id)

json.dump(config, open(file_path, "w"), indent=4)
