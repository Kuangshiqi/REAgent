#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import os
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

RE_FENCED_BLOCK = re.compile(
    r"```(?P<lang>[^\n`]*)\n(?P<code>.*?)(?:\n```|```)",
    re.DOTALL,
)

RE_PY_START = re.compile(r"```python\b[^\n]*\n", re.IGNORECASE)
RE_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)


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
                api_key='',
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


def call_model(prompt: str, model: str, max_retries: int = 60) -> str:
    m = (model or "qwen").strip().lower()
    if m == "qwen":
        return call_qwen(prompt, max_retries=max_retries)
    if m == "deepseek":
        return call_deepseek(prompt, max_retries=max_retries)
    raise ValueError(f"Unsupported model: {model}")


def build_prompt(issue_text: str, requirement_text: str, code_block: str) -> str:
    """Build prompt (intentionally matches swebench/repro2.py; do not change)."""
    PROMPT_HEADER = f"""
You are an excellent test case researcher. Now, please write a Python test file based on a repository-level code issue, a requirements document summarizing that issue, and some related code. 

The file should include multiple test cases: one part is reproduction tests, used to reproduce the bug raised in the issue and check if the bug is correctly resolved after applying the patch; the other part is regression tests, used to check if other originally correct functions still work correctly after applying the patch. These two parts of test cases should be written in the same file, using `assert` statements for judgment. The overall judgment logic has three cases: 1. If the reproduction test successfully reproduces the bug and all regression tests pass, output \"Reproduced and Regression\"; 2. If both the reproduction test and regression test pass, output \"Resolved and Regression\"; 3. Otherwise, output \"Error\".

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
            assert np.allclose(col1_floats, [12.3, 7890.0]), \"Test failed for col1\"
            assert np.allclose(col2_floats, [456.0, 10100.0]), \"Test failed for col2\"
            print(\"Issue resolved\")
        except ValueError as e:
            print(\"Issue reproduced\")
            raise AssertionError(\"Conversion failed: 'D' exponents not handled correctly\")
    except Exception as e:
        print(\"Other issues\")
        raise AssertionError(e)

test_ascii_table_with_d_exponent()
```
The required information is as follows:
<issue>
{issue_text}
</issue>

<requirement>
{requirement_text}
</requirement>

<code_files>
"""

    PROMPT_FOOTER = """
</code_files>
Test file requirements are as follows:

* The generated test file may include both reproduction tests and regression tests.
* Multiple test functions can be generated, but it's best to strictly categorize them into three cases.
* The test file must contain judgments for all three cases simultaneously, and must provide a corresponding output print after execution.
* Please output your chain-of-thought enclosed in <think>...</think>.
* Please output the code in the ```python ... ```.
"""
    return PROMPT_HEADER + (code_block or "") + PROMPT_FOOTER


def _after_think(text: str) -> str:
    if not text:
        return ""
    m = RE_THINK_CLOSE.search(text)
    if not m:
        return ""
    return text[m.end() :]


def extract_code_from_model_output(text: str) -> Tuple[str, str]:
    """Extract python code from model output.

    Policy used here (robust but simple):
    1) Only consider content after </think> (if present); if absent, fall back to whole text.
    2) Prefer the longest ```python ...``` fenced block.
    3) If only an opening ```python exists (no closing fence), take from it to EOF.
    4) Else fall back to the longest fenced code block.
    """
    if not text:
        return "", "empty"

    after = _after_think(text)
    candidate_text = after if after else text

    fenced = [
        (m.group("lang").strip().lower(), (m.group("code") or "").strip("\n"))
        for m in RE_FENCED_BLOCK.finditer(candidate_text)
    ]

    py_closed = [code for lang, code in fenced if lang == "python" and code]
    if py_closed:
        best = max(py_closed, key=len)
        return best, "python_fenced_longest"

    py_open = [m for m in RE_PY_START.finditer(candidate_text)]
    if py_open:
        # take the last opening fence to EOF (usually best)
        last = py_open[-1]
        return candidate_text[last.end() :].strip("\n"), "python_open_to_eof"

    if fenced:
        best_any = max((code for _, code in fenced), key=len)
        return best_any, "fenced_any_longest"

    return candidate_text.strip("\n"), "raw_fallback"


def create_patch_from_code(python_code: str) -> str:
    patch_header = """diff --git a/reproduce_bug.py b/reproduce_bug.py
new file mode 100644
index 0000000..e69de29
"""
    patch_body: List[str] = []
    patch_body.append("--- /dev/null")
    patch_body.append("+++ b/reproduce_bug.py")

    code_lines = python_code.split("\n")
    patch_body.append(f"@@ -0,0 +1,{len(code_lines)} @@")

    for line in code_lines:
        patch_body.append(f"+{line}")

    return patch_header + "\n".join(patch_body) + "\n"


def looks_like_patch(text: str) -> bool:
    if not text:
        return False
    t = text.lstrip()
    return t.startswith("diff --git ") and "+++ b/reproduce_bug.py" in t


def load_jsonl_as_map(path: str, key_field: str = "instance_id") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            k = obj.get(key_field)
            if k:
                out[str(k)] = obj
    return out


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def load_problem_statements(dataset_jsonl: str, needed_ids: Optional[set] = None) -> Dict[str, str]:
    """Load instance_id -> problem_statement from a dataset jsonl.

    If needed_ids is provided, only keep those ids (streaming scan).
    """
    mapping: Dict[str, str] = {}
    for _, obj in iter_jsonl(dataset_jsonl):
        iid = obj.get("instance_id")
        if not iid:
            continue
        if needed_ids is not None and iid not in needed_ids:
            continue
        mapping[iid] = obj.get("problem_statement", "")
    return mapping


def regenerate_error_tests(
    ours_result_jsonl: str,
    test_jsonl: str,
    output_jsonl: str,
    dataset_jsonl: str,
    requirement_dir: str,
    concurrency: int = 50,
    continue_on_error: bool = True,
    pool_scope: str = "global",
    model: str = "qwen",
) -> None:

    ours_map = load_jsonl_as_map(ours_result_jsonl)
    ours_ids = set(ours_map.keys())

    problems = load_problem_statements(dataset_jsonl, needed_ids=ours_ids)
    req_base = Path(requirement_dir)

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    test_records: List[Dict[str, Any]] = []
    iid_to_record: Dict[str, Dict[str, Any]] = {}
    for _, test_obj in iter_jsonl(test_jsonl):
        test_records.append(test_obj)
        iid = test_obj.get("instance_id")
        if iid:
            iid_to_record[iid] = test_obj

    error_indices_by_iid: Dict[str, List[int]] = {}
    prompt_by_iid: Dict[str, str] = {}

    for iid, rec in iid_to_record.items():
        ours = ours_map.get(iid)
        if not ours:
            continue
        results = ours.get("results")
        if not isinstance(results, list) or len(results) != 10:
            continue

        error_indices = [i for i, tag in enumerate(results) if tag == "Error_test"]
        if not error_indices:
            continue

        tp = rec.get("test_patch")
        if not isinstance(tp, list) or len(tp) != 10:
            raise ValueError(f"{iid}: test_jsonl.test_patch is not a 10-element list")

        # Normalize existing entries to patch format
        for i in range(10):
            if isinstance(tp[i], str) and tp[i] and (not looks_like_patch(tp[i])):
                tp[i] = create_patch_from_code(tp[i])
        rec["test_patch"] = tp

        inst_dir = req_base / iid
        try:
            req_text = (inst_dir / "requirement.txt").read_text(encoding="utf-8", errors="ignore")
            code_block = (inst_dir / "codefile.txt").read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] {iid}: missing requirement/codefile, skip regeneration: {e}")
            continue

        issue_text = problems.get(iid, "")
        prompt_by_iid[iid] = build_prompt(issue_text, req_text, code_block)
        error_indices_by_iid[iid] = error_indices

    tasks: List[Tuple[str, int]] = [
        (iid, idx) for iid, indices in error_indices_by_iid.items() for idx in indices
    ]
    instances_with_tasks = len(error_indices_by_iid)
    total_tasks = len(tasks)

    if total_tasks == 0:
        with open(output_jsonl, "w", encoding="utf-8") as fout:
            for obj in test_records:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return

    print(
        f"[INFO] regen-error-test: instances={instances_with_tasks} tasks={total_tasks} concurrency={concurrency}"
    )

    ok = 0
    failed = 0

    # Optional progress helper
    try:
        from tqdm import tqdm  # type: ignore

        tqdm_available = True
    except Exception as e:
        print(f"[WARN] tqdm not available, progress bar disabled: {e}")
        tqdm_available = False
        tqdm = None  # type: ignore

    def _regen_one(iid: str, idx: int) -> Tuple[str, int, str]:
        prompt = prompt_by_iid[iid]
        model_resp = call_model(prompt, model=model)
        code, _method = extract_code_from_model_output(model_resp)
        return iid, idx, create_patch_from_code(code)

    if pool_scope not in {"global", "instance"}:
        raise ValueError("pool_scope must be 'global' or 'instance'")

    if pool_scope == "global":
        # Global pool: overall concurrency == --concurrency across all instances.
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_regen_one, iid, idx) for iid, idx in tasks]
            iterator = as_completed(futs)
            if tqdm_available:
                iterator = tqdm(iterator, total=len(futs), desc="regen")  # type: ignore

            for fut in iterator:
                try:
                    iid, idx, patch = fut.result()
                except Exception as e:
                    print(f"[WARN] task failed: {e}")
                    failed += 1
                    if not continue_on_error:
                        raise
                    print(f"[WARN] task failed: {e}")
                    continue
                iid_to_record[iid]["test_patch"][idx] = patch
                ok += 1
    else:
        # Instance pool: keep per-instance concurrency semantics.
        for iid, indices in error_indices_by_iid.items():
            with ThreadPoolExecutor(max_workers=min(concurrency, len(indices))) as ex:
                futs = [ex.submit(_regen_one, iid, idx) for idx in indices]
                iterator = as_completed(futs)
                if tqdm_available:
                    iterator = tqdm(iterator, total=len(futs), desc=f"{iid}")  # type: ignore

                for fut in iterator:
                    try:
                        iid2, idx, patch = fut.result()
                    except Exception as e:
                        print(f"[WARN] {iid}: task failed: {e}")
                        failed += 1
                        if not continue_on_error:
                            raise
                        print(f"[WARN] {iid}: task failed: {e}")
                        continue
                    iid_to_record[iid2]["test_patch"][idx] = patch
                    ok += 1

    print(f"[INFO] regen done: ok={ok} failed={failed} total={total_tasks}")

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for obj in test_records:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("--ite", type=str)
    ap.add_argument("--dataset", type=str, default="verified", choices=["Lite", "Verified", "Pro"])
    ap.add_argument("--model", type=str, default="qwen", choices=["qwen", "deepseek"])

    args = ap.parse_args()
    last = str(int(args.ite) - 1)
    dataset = args.dataset

    ours_result = f"../Results/testresult/{args.model}/{dataset}/Compared/choose_with_score{args.ite}.jsonl"
    test_jsonl = f"../Results/testcase/{args.model}/{dataset}/test{last}.jsonl"
    output_jsonl = f"../Results/testcase/{args.model}/{dataset}/test{args.ite}.jsonl"
    dataset_jsonl = f"../Datasets/{dataset}.jsonl"
    requirement_dir = f"../Results/requirement/{args.model}/{dataset}/Compared/Requirement{args.ite}"

    regenerate_error_tests(
        ours_result_jsonl=ours_result,
        test_jsonl=test_jsonl,
        output_jsonl=output_jsonl,
        dataset_jsonl=dataset_jsonl,
        requirement_dir=requirement_dir,
        concurrency=20,
        model=args.model,
    )


if __name__ == "__main__":
    main()