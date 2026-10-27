#!/usr/bin/env python3

import os
import sys
import json
import argparse
import re
import jsonlines
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from openai import OpenAI


PATCH_HEADER = (
    "diff --git a/reproduce_bug.py b/reproduce_bug.py\n"
    "new file mode 100644\n"
    "index 0000000..e69de29\n"
    "--- /dev/null\n"
    "+++ b/reproduce_bug.py\n"
)

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
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Qwen API key. Set DASHSCOPE_API_KEY or QWEN_API_KEY.")

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            client = OpenAI(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=api_key,
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
    m = (model or "qwen").strip().lower()
    if m == "qwen":
        return call_qwen(prompt)
    if m == "deepseek":
        return call_deepseek(prompt)
    raise ValueError(f"Unsupported model: {model}")

def _assert_all_codefiles_exist(requirement_dir: str, instance_ids: List[str]) -> None:
    missing = []
    base = Path(requirement_dir)
    for iid in instance_ids:
        if not iid:
            continue
        p = base / iid / "codefile.txt"
        if not p.exists():
            missing.append(str(p))
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "\n..." if len(missing) > 20 else ""
        raise FileNotFoundError(f"Missing codefile.txt for {len(missing)} instances:\n{preview}{suffix}")


def build_prompt(
    issue_text: str,
    requirement_text: str,
    code_block: str,
    interface_text: str,
    include_interface: bool,
) -> str:
    interface_block = ""
    if include_interface:
        interface_block = "<interface>\n" + interface_text + "\n</interface>"

    PROMPT_HEADER = f"""
You are an excellent test case researcher. Now, please write a Python test file based on a repository-level code issue, requirements summarizing that issue and some related code. 

The file should include multiple test cases: one part is reproduction tests, used to reproduce the bug raised in the issue and check if the bug is correctly resolved after applying the patch; the other part is regression tests, used to check if other originally correct functions still work correctly after applying the patch. These two parts of test cases should be written in the same file, using `assert` statements for judgment. The overall judgment logic has three cases: 1. If the reproduction test successfully reproduces the bug and all regression tests pass, output "Reproduced and Regression"; 2. If both the reproduction test and regression test pass, output "Resolved and Regression"; 3. Otherwise, output "Error".

Below is a reproduction test example, which only includes reproduction tests, for your reference:
```python
import numpy as np
from astropy.io import fits
from astropy.io.fits import Column

def test_ascii_table_with_d_exponent():
    # Creating a FITS ASCII table with a column containing 'D' exponents
    data = np.rec.array([
        ('1.23D+01', '4.56D+02'),
        ('7.89D+03', '1.01D+04')
    ], dtype=[('col1', 'S10'), ('col2', 'S10')])

    col1 = Column(name='col1', format='A10', array=data['col1'])
    col2 = Column(name='col2', format='A10', array=data['col2'])
    
    hdu = fits.TableHDU.from_columns([col1, col2])
    
    try:
        # This will trigger the conversion of 'D' to 'E'
        converted_data = hdu.data
        # Attempt to convert the data to float
        try:
            col1_floats = converted_data['col1'].astype(float)
            col2_floats = converted_data['col2'].astype(float)
            assert np.allclose(col1_floats, [12.3, 7890.0]), "Test failed for col1"
            assert np.allclose(col2_floats, [456.0, 10100.0]), "Test failed for col2"
            print("Issue resolved")
        except ValueError as e:
            print("Issue reproduced")
            raise AssertionError("Conversion failed: 'D' exponents not handled correctly")
    except Exception as e:
        print("Other issues")
        raise AssertionError(e)

test_ascii_table_with_d_exponent()
```
The required information is as follows:
<issue>
{issue_text}
</issue>

<full_requirement>
{requirement_text}
</full_requirement>

{interface_block}

<code_files>
"""
    PROMPT_FOOTER = """
</code_files>
Test file requirements are as follows:

* The generated test file may include both reproduction tests and regression tests.
* Multiple test functions can be generated, but it's best to strictly categorize them into three cases.
* The test file must contain judgments for all three cases simultaneously, and must provide a corresponding output print after execution.
* Please output the code in the ```python ... ```.
"""
    return PROMPT_HEADER + (code_block or "") + PROMPT_FOOTER


def build_patch_from_code(code: str) -> str:
    code = (code or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = code.split("\n")
    hunk_header = f"@@ -0,0 +1,{len(lines)} @@\n"
    body = "".join(f"+{line}\n" for line in lines)
    return PATCH_HEADER + hunk_header + body


def convert_jsonl_code_to_patch(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instance_id = obj.get("instance_id")
            code = obj.get("test_patch", "")
            if not instance_id or not isinstance(code, str):
                continue

            out_obj = {
                "instance_id": instance_id,
                "test_patch": build_patch_from_code(code),
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


def merge_patch_jsonl_files(input_files: List[str], output_file: str) -> None:
    patch_dict: Dict[str, List[str]] = {}

    for fp in input_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                instance_id = data.get("instance_id")
                test_patch = data.get("test_patch", "")
                if not instance_id:
                    continue
                if instance_id not in patch_dict:
                    patch_dict[instance_id] = []
                patch_dict[instance_id].append(test_patch)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for instance_id, patches in patch_dict.items():
            output_data = {
                "instance_id": instance_id,
                "test_patch": patches,
            }
            f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")

def _process_one_item(item: Dict[str, Any], requirement_dir: str, model: str, use_interface: bool) -> Dict[str, Any]:
    instance_id = item.get("instance_id")
    if not instance_id:
        raise ValueError("Missing instance_id in input item")

    issue_text = item.get("problem_statement", "")
    interface = item.get("interface", "").strip() if use_interface else ""

    inst_dir = Path(requirement_dir) / instance_id
    req_text = (inst_dir / "requirement.txt").read_text(encoding="utf-8", errors="ignore")
    raw_code_block = (inst_dir / "codefile.txt").read_text(encoding="utf-8", errors="ignore")

    prompt = build_prompt(
        issue_text,
        req_text,
        raw_code_block,
        interface,
        include_interface=use_interface,
    )
    model_resp = call_model(prompt, model)

    match_1 = re.search(r"```python\s*(.*?)\s*```", model_resp, flags=re.DOTALL)
    model_resp_clean_1 = match_1.group(1).strip() if match_1 else model_resp.strip()

    return {
        "instance_id": instance_id,
        "test_patch": model_resp_clean_1,
        "model": model_resp
    }

def process_instances(
    input_jsonl: str,
    requirement_dir: str,
    output_path: str,
    concurrency: int = 50,
    model: str = "qwen",
    use_interface: bool = False,
):
    with open(input_jsonl, "r", encoding="utf-8") as inf:
        items = [json.loads(line) for line in inf.read().splitlines() if line.strip()]

    try:
        instance_ids = [it.get("instance_id") for it in items if it.get("instance_id")]
        _assert_all_codefiles_exist(requirement_dir, instance_ids)
    except Exception as e:
        print(f"[ERROR] Pre-check failed: {e}", file=sys.stderr)
        raise

    with jsonlines.open(output_path, 'w') as writer:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [
                ex.submit(_process_one_item, it, requirement_dir, model, use_interface)
                for it in items
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"instances(conc={concurrency})"):
                res = fut.result()
                writer.write(res)
    print(f"[INFO] Wrote raw model outputs to: {output_path}")


def run_pipeline_10_times(
    input_jsonl: str,
    requirement_dir: str,
    final_output_path: str,
    concurrency: int,
    model: str,
    use_interface: bool,
    runs: int = 10,
) -> None:
    final_output = Path(final_output_path)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_output_resolved = final_output.resolve()

    temp_raw_files: List[str] = []
    temp_patch_files: List[str] = []

    try:
        for i in range(runs):
            raw_path = final_output.parent / f"__tmp_raw_run{i}.jsonl"
            patched_path = final_output.parent / f"__tmp_patch_run{i}.jsonl"

            process_instances(
                input_jsonl=input_jsonl,
                requirement_dir=requirement_dir,
                output_path=str(raw_path),
                concurrency=concurrency,
                model=model,
                use_interface=use_interface,
            )
            convert_jsonl_code_to_patch(str(raw_path), str(patched_path))
            print(f"[INFO] Run {i + 1}/{runs} complete: {patched_path}")

            temp_raw_files.append(str(raw_path))
            temp_patch_files.append(str(patched_path))

        merge_patch_jsonl_files(temp_patch_files, str(final_output))
        print(f"[INFO] Final merged testcase patch jsonl: {final_output}")
    finally:
        for p in temp_raw_files + temp_patch_files:
            try:
                p_resolved = Path(p).resolve()
                if p_resolved != final_output_resolved:
                    os.remove(p)
            except FileNotFoundError:
                pass

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Verified", choices=["Lite", "Verified", "Pro"], type=str)
    parser.add_argument("--model", default="qwen", choices=["qwen", "deepseek"], type=str)
    args = parser.parse_args()

    dataset = args.dataset
    use_interface = dataset == "Pro"
    input_jsonl = f"../Datasets/{dataset}.jsonl"
    requirement_dir = f"../Results/requirement/{args.model}/{dataset}/Compared/Requirement0"
    output_path = f"../Results/testcase/{args.model}/{dataset}/test0.jsonl"

    run_pipeline_10_times(
        input_jsonl=input_jsonl,
        requirement_dir=requirement_dir,
        final_output_path=output_path,
        concurrency=20,
        model=args.model,
        use_interface=use_interface,
        runs=10,
    )

if __name__ == "__main__":
    main()
