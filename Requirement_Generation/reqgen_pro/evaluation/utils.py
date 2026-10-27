# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import os
import pandas as pd
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from pathlib import Path
import jsonlines
from datasets import load_dataset
from docker.models.containers import Container, ExecResult


def docker_exec(container: Container, command: str):
    """
    Execute a shell command inside a Docker container.

    Args:
        container: Docker container object.
        command: Shell command to execute.

    Returns:
        Tuple (return_code, output_str).
    """
    exec_result: ExecResult = container.exec_run(cmd=command)
    return_code = exec_result[0]
    output = exec_result[1].decode("utf-8")
    return return_code, output


def swebench_evaluate_harness_after(benchmark_harness_path, task_id):
    src_base = f"{benchmark_harness_path}/logs/run_evaluation/{task_id}/trae-agent"
    dst_base = f"results/{task_id}"
    json_src = f"{benchmark_harness_path}/trae-agent.{task_id}.json"
    json_dst = os.path.join(dst_base, "results.json")
    if not os.path.exists(src_base):
        print(f"Source directory does not exist: {src_base}")
        return
    for folder_name in os.listdir(src_base):
        src_folder = os.path.join(src_base, folder_name)
        dst_folder = os.path.join(dst_base, folder_name)
        if os.path.isdir(src_folder):
            os.makedirs(dst_folder, exist_ok=True)
            for file_name in os.listdir(src_folder):
                src_file = os.path.join(src_folder, file_name)
                dst_file = os.path.join(dst_folder, file_name)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
    os.makedirs(dst_base, exist_ok=True)
    if not os.path.exists(json_dst):
        shutil.copy2(json_src, json_dst)


def _write_problem_statement(instance_dir: Path, instance, ite: int = 0) -> int:
    instance_id = instance.get("instance_id", "")
    problem_content = instance.get("problem_statement", "").strip()
    requirements = instance.get("requirements", "").strip()
    interface = instance.get("interface", "").strip()
    if ite > 0: 
        MERGE_OLD_PATCH = Path(f"")
        patch_file = MERGE_OLD_PATCH / f"{instance_id}" / f"patch.patch"
        old_patch = patch_file.read_text(encoding="utf-8").strip()

        analysis_file = Path(f"")
        analysis = ""
        with jsonlines.open(analysis_file, "r") as reader:
            for row in reader: 
                if row.get("instance_id") == instance_id:
                    analysis = row.get("reason", "").strip()
                    break

        MERGE_REQ_DIR = Path(f"")
        req_file = MERGE_REQ_DIR / f"{instance_id}" / "requirement.txt"
        old_req = req_file.read_text(encoding="utf-8").strip()
        

        full_content = f"""
You are a requirement generator. You are given a repository issue with short explanation, a requirements document generated from the issue, an interface specification and an incorrect patch produced based on that requirements document. We have analyzed the incorrect patch and obtained findings about the problems in the requirements. Based on these information, please reflect on the errors in the original requirements and generate a new, corrected requirements document.

The analysis is conducted from the following three perspectives.

1. Conflict Deficiency: The patch is incorrect because the requirement incorrectly expresses the intent of the issue.
2. Omission Deficiency: The patch is incorrect because the requirement omits part of the issue's intent.
3. Ambiguity Deficiency: The requirement largely captures the intent of the issue, but the patch is incorrect because the requirements are not described in sufficient detail, leading to misinterpretation.

The error type is reported in <Conflict></Conflict>, <Omission></Omission> and <Ambiguity></Ambiguity>, while <Check></Check> contains the erroneous requirement items, the causes of the errors, and suggested corrections.

-- Issue -- :
{problem_content}

-- Short Explanation -- :
{requirements}

-- Interface Specification -- :
{interface}

-- Previous Requirement -- :
{old_req}

-- Analysis -- :
{analysis}

Format your response as shown in <format_example>, and the format and content of the requirement are also as follows. Include a THOUGHT section before your command where you explain your reasoning process.

<format_example>
THOUGHT: Your reasoning and analysis here.
<requirement>
1. Background
- Main Functionality: Describe the main purpose or capabilities of the repository.
- Main Modules: Describe the different modules corresponding to functionalities and the relationships between them.

2. Problem Overview
- Core Description: Provide a concise description of the problem.
- Problem Coverage: Indicate which functionalities and modules in the system are affected by the faulty code.

3. Steps to Reproduce (S2R)
- Preconditions: Specify the required state, data, or configuration before reproducing the issue.
- Key Conditions: Clarify under what circumstances the error occurs, including specific inputs, versions, etc..
- Reproduction Commands: Provide the complete procedure from start to triggering the issue.

4. Actual Behavior (AR)
- Erroneous Behavior: Describe the actual erroneous output or exception from the faulty module.
- Correct Behavior: Describe the normal behavior of other modules using the affected code.

5. Expected Behavior (ER)
- Ideal Behavior: Describe the expected result when the system functions correctly.
- Success Criteria: Explain how to verify that the issue has been fixed.

6. Environment
- Dependencies and Imports: List the dependencies, required versions, APIs, libraries, or modules necessary to reproduce and fix the issue.

7. Root Cause Analysis
- Error Cause: Infer the potential root cause of the issue from the observed symptoms.
- Code Paths: Point out the key functions, call chains, or logic paths where the problem originates.

8. Solution
- Modification Location: Specify the files, functions, and code snippets that need to be modified. Enclose the file path that needs to be modified within single parentheses.
- Modification Content: Provide a detailed description of the code snippets that need to be added, deleted, or modified, and explain specifically how the modifications should be made.
- Impact Scope: Explain which modules will be affected and which correct functionalities should remain unchanged.

9. Additional Notes
- Security, Compatibility, or Other Considerations: Provide additional notes that may affect future maintenance, deployment, or security.
</requirement>
</format_example>

You need to complete all tasks within 50 steps and output the content in the above format.
"""

    else:
        full_content = f"""
You are an expert AI software engineering agent with strong capabilities in requirement engineering, debugging, and code maintenance. Given an issue, a simple requirement and an interface specification, you need to generate full requirements based on these information.

-- Issue -- :
{problem_content}

-- Requirements -- :
{requirements}

-- Interface Specification -- :
{interface}

Format your response as shown in <format_example>, and the format and content of the requirement are also as follows. Include a THOUGHT section before your command where you explain your reasoning process.

<format_example>
THOUGHT: Your reasoning and analysis here.
<requirement>
1. Background
- Main Functionality: Describe the main purpose or capabilities of the repository.
- Main Modules: Describe the different modules corresponding to functionalities and the relationships between them.

2. Problem Overview
- Core Description: Provide a concise description of the problem.
- Problem Coverage: Indicate which functionalities and modules in the system are affected by the faulty code.

3. Steps to Reproduce (S2R)
- Preconditions: Specify the required state, data, or configuration before reproducing the issue.
- Key Conditions: Clarify under what circumstances the error occurs, including specific inputs, versions, etc..
- Reproduction Commands: Provide the complete procedure from start to triggering the issue.

4. Actual Behavior (AR)
- Erroneous Behavior: Describe the actual erroneous output or exception from the faulty module.
- Correct Behavior: Describe the normal behavior of other modules using the affected code.

5. Expected Behavior (ER)
- Ideal Behavior: Describe the expected result when the system functions correctly.
- Success Criteria: Explain how to verify that the issue has been fixed.

6. Environment
- Dependencies and Imports: List the dependencies, required versions, APIs, libraries, or modules necessary to reproduce and fix the issue.

7. Root Cause Analysis
- Error Cause: Infer the potential root cause of the issue from the observed symptoms.
- Code Paths: Point out the key functions, call chains, or logic paths where the problem originates.

8. Solution
- Modification Location: Specify the files, functions, and code snippets that need to be modified. Enclose the file path that needs to be modified within single parentheses.
- Modification Content: Provide a detailed description of the code snippets that need to be added, deleted, or modified, and explain specifically how the modifications should be made.
- Impact Scope: Explain which modules will be affected and which correct functionalities should remain unchanged.

9. Additional Notes
- Security, Compatibility, or Other Considerations: Provide additional notes that may affect future maintenance, deployment, or security.
</requirement>
</format_example>

You need to complete all tasks within 50 steps and output the content in the above format.
"""
    
    out_file = instance_dir / "problem_statement.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        return f.write(full_content)

def _sweap_image_name(instance_id: str) -> str:
    """
    完全使用本地 sweap 镜像，不再使用 benchmark 默认镜像规则
    期望格式: instance_{repo_base}__{repo_name}-{hash}
    """
    if not instance_id.startswith("instance_"):
        raise ValueError(f"Invalid instance_id: {instance_id}")

    body = instance_id[len("instance_"):]
    if "__" not in body:
        raise ValueError(f"Invalid instance_id: {instance_id}")

    repo_base, rest = body.split("__", 1)
    if "-" not in rest:
        raise ValueError(f"Invalid instance_id: {instance_id}")

    repo_name, hash_part = rest.split("-", 1)

    tag = f"{repo_base}.{repo_name}-{repo_base}__{repo_name}-{hash_part}"
    if len(tag) > 128:
        tag = tag[:128]

    return f"jefzda/sweap-images:{tag}"

@dataclass
class BenchmarkConfig:
    valid_datasets: list[str]
    load_dataset: Callable[[str], Any]
    image_name: Callable[[str], str]
    problem_statement: Callable[..., Any]
    working_dir: Callable[[str], str]
    evaluate_harness: Callable[..., list[str]]
    evaluate_harness_before: Callable[..., Any]
    evaluate_harness_after: Callable[..., Any]


BENCHMARK_CONFIG: dict[str, BenchmarkConfig] = {
    # SWE-bench
    "SWE-bench": BenchmarkConfig(
        valid_datasets=["SWE-bench_Pro"],
        load_dataset=lambda dataset_name: pd.read_parquet(
            "../../../Datasets/Pro.parquet"
        ).to_dict(orient="records"),
        image_name=_sweap_image_name,
        problem_statement=lambda instance, instance_dir, ite=0: (
            _write_problem_statement(instance_dir, instance, ite)
        ),
        working_dir=lambda instance_id: "/app/",
        evaluate_harness=lambda dataset_name, task_results_dir, task_id, max_workers: [
            "swebench_venv/bin/python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            f"princeton-nlp/{dataset_name}",
            "--predictions_path",
            (task_results_dir / "predictions.json").absolute().as_posix(),
            "--max_workers",
            str(max_workers),
            "--run_id",
            task_id,
            "--cache_level",
            "instance",
            "--instance_image_tag",
            "latest",
        ],
        evaluate_harness_before=lambda *args, **kwargs: None,
        evaluate_harness_after=swebench_evaluate_harness_after,
    ),
}
