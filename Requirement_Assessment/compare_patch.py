import json
import os
import argparse
from tqdm import tqdm
import concurrent.futures
from openai import OpenAI
from typing import Optional
import time

PATCH_DIRS = []
CODEFILE_DIRS = []
RESULT_FILES = []
N = 0

score_map = {}
results_map = {}
issue_map = {}
final_results = []
other_reason_map = {}

def call_deepseek(prompt: str, max_retries: int = 60) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key="",
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


def read_patch(instance_id, patch_idx):
    patch_path = os.path.join(PATCH_DIRS[patch_idx], str(instance_id), f"patch.patch")
    try:
        with open(patch_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: failed to read patch for {instance_id} at {patch_path}: {e}")
        return ""

def read_codefile(instance_id, patch_idx):
    path = os.path.join(CODEFILE_DIRS[patch_idx], str(instance_id), "codefile.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: failed to read codefile for {instance_id} at {path}: {e}")
        return ""


def init_runtime(ite: str, last_value: str, dataset: str):
    global now, last, PATCH_DIRS, CODEFILE_DIRS, RESULT_FILES, N
    global score_map, results_map, issue_map, final_results, other_reason_map

    now = str(ite)
    last = str(last_value)

    PATCH_DIRS = [
        f"../Results/patch/{model}/{dataset}/Compared/patch{last}",
        f"../Results/patch/{model}/{dataset}/Generated/patch{now}",
    ]
    CODEFILE_DIRS = [
        f"../Results/requirement/{model}/{dataset}/Compared/Requirement{last}",
        f"../Results/requirement/{model}/{dataset}/Generated/Requirement{now}",
    ]

    RESULT_FILES = [f"../Results/testresult/{model}/{dataset}/Compared/result{last}.jsonl", f"../Results/testresult/{model}/{dataset}/Generated/result{now}.jsonl"]
    N = len(PATCH_DIRS)
    assert len(CODEFILE_DIRS) == N and len(RESULT_FILES) == N

    score_map = {}
    results_map = {}
    for idx, fname in enumerate(RESULT_FILES):
        result_path = os.path.join(fname)
        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"Warning: failed to parse line in {fname}: {line[:30]}... Error: {e}")
                    continue
                iid = data.get("instance_id")
                if not iid:
                    continue
                score_map[(iid, idx)] = data.get("avgpassratio", 0.0)
                results_map[(iid, idx)] = data.get("results", [])

    dataset_map = {
        "Lite": "Lite.jsonl",
        "Verified": "verified.jsonl",
        "Pro": "Pro.jsonl",
    }
    issue_path = f"../Datasets/{dataset_map.get(dataset, 'Pro.jsonl')}"
    issue_map = {}
    with open(issue_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            issue_map[data["instance_id"]] = data["problem_statement"]

    with open(RESULT_FILES[0], "r", encoding="utf-8") as f:
        final_results = [json.loads(line) for line in f]

    other_reason_map = {}
    feedback_candidates = [f"../Results/feedback/{model}/{dataset}/feedback{last}.jsonl"]
    for fp in feedback_candidates:
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception as e:
                    print(f"Warning: failed to parse feedback line in {fp}: {line[:30]}... Error: {e}")
                    continue
                iid = d.get("instance_id")
                if iid and "reason" in d:
                    other_reason_map[iid] = d.get("reason")


def make_result(iid, patch, file, avg, idx):
    res = {
        "instance_id": iid,
        "patch": patch,
        "file": file,
        "avgpassratio": avg,
    }
    if file != RESULT_FILES[-1]:
        if iid in other_reason_map and other_reason_map[iid] is not None:
            res["nousereason"] = other_reason_map[iid]

    results_list = results_map.get((iid, idx))
    res["results"] = results_list if isinstance(results_list, list) else []
    return res


def build_choose_prompt(issue_text, candidates, shared_code=None):
    body = ""
    if shared_code is not None:
        for idx, (tag, patch) in enumerate(candidates.items(), 1):
            body += f"\n[{tag}]\n<patch{idx}>\n{patch}\n</patch{idx}>\n"
        shared_block = f"<code_files>\n{shared_code}\n</code_files>\n"
    else:
        for idx, (tag, (patch, code)) in enumerate(candidates.items(), 1):
            body += f"\n[{tag}]\n<patch{idx}>\n{patch}\n</patch{idx}>\n<code_files>\n{code}\n</code_files>\n"
        shared_block = ""

    prompt = f"""
You are given a repository-level issue, two patches generated according to the issue and several code files related with this issue.
Please select the most correct patch that best addresses the bug raised in the issue.

<issue>
{issue_text}
</issue>

{shared_block}

<candidates>
{body}
</candidates>

Your output must be exactly:
<choose>patch1/patch2</choose>
""".strip()
    return prompt

def has_error_patch(iid: str, idx: int) -> bool:
    r = results_map.get((iid, idx))
    return isinstance(r, list) and ("Error_Patch" in r)


def call_model(prompt: str) -> str:
    if model == "qwen":
        return call_qwen(prompt)
    if model == "deepseek":
        return call_deepseek(prompt)
    raise ValueError(f"Unsupported model: {model}")

def process_instance(row):
    instance_id = row["instance_id"]
    issue_text = issue_map.get(instance_id, "")
    file_name = row.get("file") or RESULT_FILES[0]

    idx1 = None
    for i in range(N - 1):
        if file_name == RESULT_FILES[i]:
            idx1 = i
            break
    if idx1 is None:
        idx1 = 0
    idx2 = N - 1

    patch1 = row.get("patch")
    if patch1 is None:
        patch1 = read_patch(instance_id, idx1)

    file1 = file_name
    score1 = score_map.get((instance_id, idx1), 0.0)
    code1 = read_codefile(instance_id, idx1)

    patch2 = read_patch(instance_id, idx2)
    code2 = read_codefile(instance_id, idx2)
    score2 = score_map.get((instance_id, idx2), None)
    file2 = RESULT_FILES[idx2]

    if score2 is None or patch2.strip() == "":
        return make_result(instance_id, patch1, file1, score1, idx1)

    err1 = has_error_patch(instance_id, idx1)
    err2 = has_error_patch(instance_id, idx2)

    if err1 and not err2:
        return make_result(instance_id, patch2, file2, score2, idx2)
    elif err2 and not err1:
        return make_result(instance_id, patch1, file1, score1, idx1)

    if score1 > score2:
        return make_result(instance_id, patch1, file1, score1, idx1)
    elif score2 > score1:
        return make_result(instance_id, patch2, file2, score2, idx2)

    if code1 == code2:
        shared_code = code1
        candidates = {"patch1": patch1, "patch2": patch2}
        prompt = build_choose_prompt(issue_text, candidates, shared_code=shared_code)
    else:
        candidates = {"patch1": (patch1, code1), "patch2": (patch2, code2)}
        prompt = build_choose_prompt(issue_text, candidates, shared_code=None)

    resp = call_model(prompt)
    if "<choose>patch1</choose>" in resp:
        choose = 1
    elif "<choose>patch2</choose>" in resp:
        choose = 2
    else:
        choose = 1

    if choose == 1:
        return make_result(instance_id, patch1, file1, score1, idx1)
    else:
        return make_result(instance_id, patch2, file2, score2, idx2)

def main():
    global model

    parser = argparse.ArgumentParser()
    parser.add_argument("--ite", type=str)
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "deepseek"])
    parser.add_argument("--dataset", type=str, default="Pro", choices=["Lite", "Verified", "Pro"])
    args = parser.parse_args()

    model = args.model
    last_value = str(int(args.ite) - 1)
    init_runtime(args.ite, last_value, args.dataset)

    output_file = f"../Results/testresult/{args.model}/{args.dataset}/Compared/choose_with_score{args.ite}.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_instance, row) for row in final_results]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Comparing"):
                result = future.result()
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()