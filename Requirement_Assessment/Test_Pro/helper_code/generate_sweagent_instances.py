import argparse
import os
import yaml
import json
from tqdm import tqdm

from image_uri import get_dockerhub_image_uri
from create_problem_statement import create_problem_statement


def load_local_dataset(path):
    """Load local SWE-bench-Pro style jsonl"""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate_instances_local(dockerhub_username, jsonl_path):
    print(f"Loading local dataset from {jsonl_path}")
    rows = load_local_dataset(jsonl_path)

    instances = []
    print(f"Processing {len(rows)} instances...")

    for row in tqdm(rows):
        instance_id = row["instance_id"]
        repo_name = row.get("repo", "")

        image_name = get_dockerhub_image_uri(
            instance_id,
            dockerhub_username,
            repo_name
        )

        problem_statement = create_problem_statement(row)

        instance = {
            "image_name": image_name,
            "problem_statement": problem_statement,
            "instance_id": instance_id,
            "base_commit": row["base_commit"],
            "repo_name": "app"
        }

        instances.append(instance)

    return instances


def write_yaml(instances, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(instances, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"✓ Wrote {len(instances)} instances to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dockerhub_username", required=True)
    parser.add_argument("--input_jsonl", required=True, help="Your sampled pro.jsonl")
    parser.add_argument(
        "--output_path",
        default="SWE-agent/data/instances.yaml"
    )
    args = parser.parse_args()

    instances = generate_instances_local(
        args.dockerhub_username,
        args.input_jsonl
    )

    write_yaml(instances, args.output_path)


if __name__ == "__main__":
    main()
