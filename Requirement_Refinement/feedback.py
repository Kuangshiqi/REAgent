#!/usr/bin/env python3
import argparse
import json
import random
import re
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import jsonlines
from tqdm import tqdm
from openai import OpenAI


def call_deepseek(prompt: str, max_retries: int = 60) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key='',
                base_url="https://api.deepseek.com/v1"
            )
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.5,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"an error occurred: {e} in attempt {attempt + 1}/{max_retries}")
            last_err = e
            time.sleep(10)
    raise last_err if last_err else RuntimeError("call_deepseek failed")


def call_qwen(prompt: str, max_retries: int = 60) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            client = OpenAI(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key="",
            )

            completion = client.chat.completions.create(
                extra_body={"reasoning": {"enabled": True}},
                model="qwen-plus-latest",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.5,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"an error occurred: {e} in attempt {attempt + 1}/{max_retries}")
            last_err = e
            time.sleep(10)
    raise last_err if last_err else RuntimeError("call_qwen failed")


def call_model(prompt: str, model: str) -> str:
    if model == "qwen":
        return call_qwen(prompt)
    if model == "deepseek":
        return call_deepseek(prompt)
    raise ValueError(f"Unsupported model: {model}")



def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


ResultVal = Union[bool, str]


def _pick_testcases_from_lists(results_list: List[ResultVal], test_patch_list: List[str]) -> List[str]:
    n = min(len(results_list), len(test_patch_list))
    if n == 0:
        return []

    false_cases = [test_patch_list[i] or "" for i in range(n) if results_list[i] is False]
    if false_cases:
        return [c for c in false_cases if c.strip()]

    non_true_cases = [test_patch_list[i] or "" for i in range(n) if results_list[i] is not True]
    return [c for c in non_true_cases if c.strip()]


def evaluate_requirements(
    input_jsonl: str,
    requirements_dir: str,
    output_path: str,
    testcase_jsonl: str,
    concurrency: int = 50,
    result_jsonl: str = None,
    model: str = "deepseek",
):
    req_base = Path(requirements_dir)
    patch_map: Dict[str, str] = {}
    nousereason_map: Dict[str, str] = {}
    results_map: Dict[str, List[ResultVal]] = {}
    if result_jsonl:
        try:
            with jsonlines.open(result_jsonl, "r") as reader:
                payload = list(reader)
        except Exception as e:
            raise RuntimeError(f"Failed to load result jsonl: {result_jsonl}, error: {e}")

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                iid = item.get("instance_id")
                if not iid:
                    continue
                patch_text = item.get("patch", "") or ""
                patch_map[iid] = patch_text
                if "results" in item and item.get("results") is not None:
                    results_map[iid] = item.get("results") or []
                if "nousereason" in item and item.get("nousereason") is not None:
                    nousereason_map[iid] = item.get("nousereason")

    with jsonlines.open(input_jsonl, "r") as reader:
        lines = list(reader)
    input_ids = [item.get("instance_id") for item in lines if "instance_id" in item]

    if not results_map:
        for item in lines:
            iid = item.get("instance_id")
            if not iid:
                continue
            if "results" in item and item.get("results") is not None:
                results_map[iid] = item.get("results") or []

    problem_map: Dict[str, str] = {
        item.get("instance_id"): item.get("problem_statement", "") for item in lines
    }

    with jsonlines.open(testcase_jsonl, "r") as tester:
        testcase_lines = list(tester)
    testcase_map: Dict[str, List[str]] = {
        item.get("instance_id"): (item.get("test_patch") or []) for item in testcase_lines
    }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_one(iid: str) -> Dict[str, Any]:
        req_file = req_base / iid / "requirement.txt"
        patch_text = patch_map.get(iid, "")

        requirement_text = load_text_file(req_file) if req_file.exists() else ""

        if patch_text.strip() == "":
            return {
                "instance_id": iid,
                "reason": "EmptyPatch!",
            }

        problem_statement = problem_map.get(iid, "")

        results_list = results_map.get(iid, [])
        t_list = testcase_map.get(iid, [])
        testcases = _pick_testcases_from_lists(results_list, t_list)
        testcase = "\n\n".join([f"[TestCase {i+1}]\n{tc}" for i, tc in enumerate(testcases)])

        extra_nousereason = nousereason_map.get(iid, "")
        prompt = f"""
I will provide you with an issue from a repository, a requirement summarized based on the issue and an incorrect patch with test patch. 
The format of the requirements document is as follows:
<requirement>
1. Background
- Main Functionality: Describe the main purpose or capabilities of the repository.
- Main Modules: Describe the different modules corresponding to functionalities and the relationships between them.

2. Problem Overview
- Core Description: Provide a concise description of the problem.
- Problem Coverage: Indicate which functionalities and modules in the system are affected by the faulty code.

3. Steps to Reproduce
- Preconditions: Specify the required state, data, or configuration before reproducing the issue.
- Key Conditions: Clarify under what circumstances the error occurs, including specific inputs, versions, etc..
- Reproduction Commands: Provide the complete procedure from start to triggering the issue.

4. Actual Behavior
- Erroneous Behavior: Describe the actual erroneous output or exception from the faulty module.
- Correct Behavior: Describe the normal behavior of other modules using the affected code.

5. Expected Behavior
- Ideal Behavior: Describe the expected result when the system functions correctly.
- Success Criteria: Explain how to verify that the issue has been fixed.

6. Environment
- Dependencies and Imports: List the dependencies, required versions, APIs, libraries, or modules necessary to reproduce and fix the issue.

7. Root Cause Analysis
- Error Cause: Infer the potential root cause of the issue from the observed symptoms.
- Code Paths: Point out the key functions, call chains, or logic paths where the problem originates.

8. Solution
- Modification Location: Specify the files, functions, and code snippets that need to be modified.
- Modification Content: Provide a detailed description of the code snippets that need to be added, deleted, or modified, and explain specifically how the modifications should be made.
- Impact Scope: Explain which modules will be affected and which correct functionalities should remain unchanged.

9. Additional Notes
- Security, Compatibility, or Other Considerations: Provide additional notes that may affect future maintenance, deployment, or security.
</requirement>

Now, based on these information, please review the quality of the requirement by analyzing the incorrect patch from the following three perspectives.

1. Conflict Reficiency(<Conflict>): The patch is incorrect because the requirement wrongly expresses the intent of the issue.
2. Omission Reficiency(<Omission>): The patch is incorrect because the requirement is missing part of the issue's intent.
3. Ambiguity Reficiency(<Ambiguity>): The requirement generally reflects the issue's intent correctly, but the patch is incorrect because the requirements are not sufficiently detailed, which leads to misinterpretation.

Multiple of the above cases may exist simultaneously. Please first determine whether each of the three causes is present. Then, in <Check></Check>, output the Conflict Reficiency items, the reasons for the errors, and suggestions for correction. Each item must correspond exactly to one of the nine requirements listed above and use the same numbering. If you believe certain requirement items are already correct, do not include them.

For Conflict Reficiency, emphasize how the requirement is incorrect and the direction for correction.
For Omission Reficiency, emphasize what content needs to be added.
For Ambiguity Reficiency, emphasize where the patch misinterpreted the requirement and how the requirement should be refined.

The output format should be as follows:

<Conflict>True/False</Conflict>
<Omission>True/False</Omission>
<Ambiguity>True/False</Ambiguity>
<Check>
1. Background: ...
3. Steps to Reproduce: ...
7. Root Cause Analysis: ...
</Check>

All information is provided below:

-- Issue --:
{problem_statement}

-- Requirement --:
{requirement_text}

-- Incorrect Patch --:
{patch_text}

-- Test Patch --:
{testcase}

"""
        if extra_nousereason:
            print("nouse")
            prompt += f"""
Note that the following modifications have been proven incorrect; please try a different approach.

-- Unuseful Analysis --
{extra_nousereason}
"""

        model_output = call_model(prompt, model)

        return {
            "instance_id": iid,
            "reason": model_output,
        }

    with jsonlines.open(output_path, "w") as writer:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_process_one, iid): iid for iid in input_ids}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"LLM judging (concurrency={concurrency})"):
                writer.write(fut.result())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ite", type=str)
    parser.add_argument("--model", type=str, default="deepseek", choices=["qwen", "deepseek"])
    parser.add_argument("--dataset", type=str, default="Verified", choices=["Lite", "Verified", "Pro"])
    args = parser.parse_args()
    
    input_file = f"../Datasets/{args.dataset}.jsonl"
    requirements_dir = f"../Results/requirement/{args.model}/{args.dataset}/Compared/Requirement{args.ite}"
    result_jsonl = f"../Results/testresult/{args.model}/{args.dataset}/Compared/choose_with_score{args.ite}.jsonl"
    output_jsonl = f"../Results/feedback/{args.model}/{args.dataset}/feedback{args.ite}.jsonl"
    testcase_jsonl = f"../Results/testcase/{args.model}/{args.dataset}/test{args.ite}.jsonl"

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    evaluate_requirements(
        input_jsonl=input_file,
        requirements_dir=requirements_dir,
        output_path=output_jsonl,
        testcase_jsonl=testcase_jsonl,
        concurrency=20,
        result_jsonl=result_jsonl,
        model=args.model,
    )


if __name__ == "__main__":
    main()
