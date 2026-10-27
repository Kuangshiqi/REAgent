#!/usr/bin/env python3

import os
import json
import argparse
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from rank_bm25 import BM25Okapi
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from git import Repo
import tiktoken
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
                    {"role": "system", "content": "You are a patches generator."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=0.1,
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
                temperature=0.1,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"an error occurred: {e} in attempt {attempt + 1}/{max_retries}")
            last_err = e
            time.sleep(10)
    raise last_err if last_err else RuntimeError("call_qwen failed")



CODE_EXT_WHITELIST_LITE = {'.py'}

CODE_EXT_WHITELIST_PRO = {'.py', '.c', '.cpp', '.h', '.hpp', '.java', '.js', '.ts', '.jsx', '.tsx',
                          '.go', '.rb', '.rs', '.sh', '.md', '.rst', '.txt', '.json', '.yml', '.yaml', '.cs'}

VALID_REPO_SLUGS_LITE = {
    "astropy",
    "django",
    "flask",
    "matplotlib",
    "pylint",
    "pytest",
    "requests",
    "scikit-learn",
    "seaborn",
    "sphinx",
    "sympy",
    "xarray"
}

VALID_REPO_SLUGS_PRO = {
    "flipt",
    "navidrome",
    "teleport",
    "tutanota",
    "vuls",
    "webclients"
}

CODE_EXT_WHITELIST = set(CODE_EXT_WHITELIST_LITE)
VALID_REPO_SLUGS = set(VALID_REPO_SLUGS_LITE)
_EXT_ALTS = "|".join(re.escape(e.lstrip('.')) for e in CODE_EXT_WHITELIST)
RE_PATH_IN_BRACKETS = re.compile(
    rf"(?:\(|\[|\{{)\s*([^()\[\]{{}}\n\r]*?\.(?:{_EXT_ALTS}))\s*(?:\)|\]|\}})",
    re.IGNORECASE,
)
RE_MODIFICATION_LOCATION_BLOCK = re.compile(
    r"Modification Location[:\s\-\#]*(.*?)(?=Modification Content)",
    re.DOTALL | re.IGNORECASE,
)
RE_ANY_QUOTED = re.compile(
    r"(?:"
    r"`\s*'([^'\n\r]+?)'\s*`"
    r"|`\s*\"([^\"\n\r]+?)\"\s*`"
    r"|`([^`\n\r]+?)`"
    r"|'([^'\n\r]+?)'"
    r"|\"([^\"\n\r]+?)\""
    r")"
)
TRAILING_PUNCT = ".,;:!?"


def configure_dataset_rules(dataset: str):
    global CODE_EXT_WHITELIST, VALID_REPO_SLUGS, _EXT_ALTS, RE_PATH_IN_BRACKETS
    ds = (dataset or "").strip()
    if ds in {"Lite", "Verified"}:
        CODE_EXT_WHITELIST = set(CODE_EXT_WHITELIST_LITE)
        VALID_REPO_SLUGS = set(VALID_REPO_SLUGS_LITE)
    elif ds == "Pro":
        CODE_EXT_WHITELIST = set(CODE_EXT_WHITELIST_PRO)
        VALID_REPO_SLUGS = set(VALID_REPO_SLUGS_PRO)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    _EXT_ALTS = "|".join(re.escape(e.lstrip('.')) for e in CODE_EXT_WHITELIST)
    RE_PATH_IN_BRACKETS = re.compile(
        rf"(?:\(|\[|\{{)\s*([^()\[\]{{}}\n\r]*?\.(?:{_EXT_ALTS}))\s*(?:\)|\]|\}})",
        re.IGNORECASE,
    )


def _has_valid_ext(s: str) -> bool:
    if not s:
        return False
    s = s.lower()
    for ext in CODE_EXT_WHITELIST:
        if ext and s.endswith(ext.lower()):
            return True
    return False


def _strip_testbed_prefix(s: str) -> str:
    s = (s or "").strip().replace("\\", "/")
    for marker in ("testbed/", "app/"):
        idx = s.lower().find(marker)
        if idx != -1:
            return s[idx + len(marker):]
    return s


def _clean_path_like(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in "\"'`"):
        s = s[1:-1].strip()
    if s and s[0] in "([{" and s[-1] in ")]}":
        s = s[1:-1].strip()
    while s and s[-1] in TRAILING_PUNCT:
        s = s[:-1]
    s = s.strip().replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    return s


def extract_nested_quoted_strings(text: str, max_rounds: int = 20) -> List[str]:
    if not text:
        return []
    seen = set()
    ordered: List[str] = []
    frontier = [text]
    rounds = 0
    while frontier and rounds < max_rounds:
        rounds += 1
        next_frontier = []
        produced_any = False
        for chunk in frontier:
            for m in RE_ANY_QUOTED.finditer(chunk):
                s = next((g for g in m.groups() if g), None)
                if not s:
                    continue
                s = s.strip()
                if s not in seen:
                    seen.add(s)
                    ordered.append(s)
                    next_frontier.append(s)
                    produced_any = True
        if not produced_any:
            break
        frontier = next_frontier
    return ordered


def extract_paths_from_brackets(text: str) -> List[str]:
    if not text:
        return []
    out: List[str] = []
    seen = set()
    for m in RE_PATH_IN_BRACKETS.finditer(text):
        s = _strip_testbed_prefix(_clean_path_like(m.group(1)))
        if _has_valid_ext(s) and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def candidate_paths_from_requirement(issue_text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for s in extract_nested_quoted_strings(issue_text):
        s2 = _strip_testbed_prefix(_clean_path_like(s))
        if _has_valid_ext(s2) and s2 not in seen:
            seen.add(s2)
            out.append(s2)
    for s2 in extract_paths_from_brackets(issue_text):
        if s2 not in seen:
            seen.add(s2)
            out.append(s2)
    return out


def write_pathtext_from_requirement(inst_dir: Path) -> bool:
    req_file = inst_dir / "requirement.txt"
    pathtext_file = inst_dir / "pathtext.txt"
    if not req_file.exists():
        return False
    try:
        content = req_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    match = RE_MODIFICATION_LOCATION_BLOCK.search(content)
    if not match:
        return False
    extracted_text = (match.group(1) or "").strip()
    if not extracted_text:
        return False
    pathtext_file.write_text(extracted_text, encoding="utf-8")
    return True

def get_local_repo_path(repo_full_name: str, base_dir: str = "../repos") -> str:
    repo_slug = (repo_full_name or "").split("/")[-1]
    if repo_slug not in VALID_REPO_SLUGS:
        raise FileNotFoundError(f"repo slug '{repo_slug}' 不在允许列表：{sorted(VALID_REPO_SLUGS)}")
    dest = Path(base_dir) / repo_slug
    if not (dest.exists() and (dest / ".git").exists()):
        raise FileNotFoundError(f"本地仓库 {dest} 不存在，请先手动克隆。")
    return str(dest)

def checkout_commit(repo_path: str, commit_sha: Optional[str] = None):
    repo = Repo(repo_path)
    if not commit_sha:
        return
    try:
        repo.git.reset("--hard")
        repo.git.clean("-fd")
        repo.git.checkout(commit_sha)
    except Exception as e:
        print(f"无法 checkout {commit_sha} in {repo_path}: {e}")

def approx_token_count(text: str) -> int:
    words = re.findall(r"\S+", text)
    return max(1, int(len(words) * 1.3))

def count_tokens(text: str) -> int:
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("gpt2")
            return len(enc.encode(text))
        except Exception:
            return approx_token_count(text)
    else:
        return approx_token_count(text)

def simple_tokenize(text: str):
    toks = re.findall(r"[A-Za-z_]\w+|[0-9]+|[^\s\w]", text)
    return [t.lower() for t in toks if t.strip()]

def list_repo_files(repo_root: str, exts=None):
    exts = exts if exts else CODE_EXT_WHITELIST
    files = []
    for p in Path(repo_root).rglob("*"):
        if p.is_file() and (not p.suffix or p.suffix.lower() in exts):
            try:
                files.append(str(p))
            except Exception:
                continue
    return files

PATCH_DIFF_RE = re.compile(r"^diff --git a\/(?P<a>[^ ]+) b\/(?P<b>[^ ]+)", re.MULTILINE)
HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def fix_patch_text(patch_text: str) -> str:
    lines = patch_text.splitlines()
    out = []

    i = 0
    while i < len(lines):
        line = lines[i]
        m = HUNK_RE.match(line)

        if not m:
            out.append(line)
            i += 1
            continue

        old_start = int(m.group(1))
        new_start = int(m.group(3))

        old_count = 0
        new_count = 0

        hunk_lines = []
        i += 1

        while i < len(lines):
            l = lines[i]
            if l.startswith('@@ ') or l.startswith('diff --git'):
                break
            hunk_lines.append(l)
            i += 1

        for l in hunk_lines:
            if l.startswith('-') and not l.startswith('---'):
                old_count += 1
            elif l.startswith('+') and not l.startswith('+++'):
                new_count += 1
            elif l.startswith('\\'):
                continue
            else:
                old_count += 1
                new_count += 1

        fixed_header = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@"
        out.append(fixed_header)
        out.extend(hunk_lines)

    return "\n".join(out) + "\n"


def parse_patch_file_paths(patch_text: str) -> List[str]:
    matches = PATCH_DIFF_RE.findall(patch_text)
    paths, seen = [], set()
    for a, b in matches:
        p = b if b else a
        if p not in seen:
            seen.add(p)
            paths.append(p)
    return paths

def build_prompt(issue_text: str, code_block: str, interface: str = "") -> str:
    interface_block = ""
    if interface and interface.strip():
        interface_block = f"""
<interface>
{interface}
</interface>
"""

    PROMPT_HEADER = f"""
You will be given an issue resolution requirement and several related code files. Based on the requirement and the provided code, you should write a patch to resolve the issue. The patch must be applied only within the scope of the given files.
<requirement>
{issue_text}
</requirement>
{interface_block}

<code_files>
"""
    PROMPT_FOOTER = """
</code_files>
Here is an example of a patch file. It consists of changes to the code base. It specifies the file names, the line numbers of each change, and the removed and added lines. A single patch file can contain changes to multiple files.
```patch
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,5 +1,5 @@
 def add(a, b):
-    return a + b
+    return a - b

 def mul(a, b):
     return a * b
```

* Please align the output precisely with the original code, and ensure the context lines (starting with ' ') are an exact match for the target repository's code at the time of the change.
* Ensure the file paths in the diff --git lines are relative to the repository root (e.g., django/forms/widgets.py) and use the a/ and b/ prefixes correctly.
* Please output your briefly chain-of-thought enclosed in <think>...</think>.
* Please output the patch in the ```patch ... ```.
"""
    return PROMPT_HEADER + (code_block or "") + PROMPT_FOOTER

def build_code_block(retrieved_files: List[Tuple[str, str]]) -> str:
    code_block = ""
    for p, c in retrieved_files:
        code_block += f"[start of {p}]\n{c}\n[end of {p}]\n"
    return code_block

MAX_CONTEXT_TOKENS = 13000

def build_bm25_index(files: List[str], repo_root: str):
    docs = []
    for f in files:
        rel_path = os.path.relpath(f, repo_root)
        try:
            content = open(f, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            content = ""
        text = f"{rel_path}\n{content}"
        toks = simple_tokenize(text) or [""]
        docs.append(toks)
    return BM25Okapi(docs)

def sparse_select_files(repo_root: str, query: str, max_tokens: int = MAX_CONTEXT_TOKENS):
    files = list_repo_files(repo_root)
    if not files:
        return []

    files = [p for p in files if not os.path.basename(p).lower().startswith("readme")]

    bm25 = build_bm25_index(files, repo_root=repo_root)
    q_toks = simple_tokenize(query)
    scores = bm25.get_scores(q_toks)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    selected, total_tokens = [], 0

    def read_file(p):
        try:
            return open(p, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            return ""

    for idx in ranked_idx:
        if total_tokens >= max_tokens:
            break

        p = files[idx]
        content = read_file(p)
        if not content:
            continue

        remaining = max_tokens - total_tokens
        tok = count_tokens(content)

        rel_path = os.path.relpath(p, repo_root)

        if tok <= remaining:
            selected.append((rel_path, content))
            total_tokens += tok
            continue
        continue

    return selected


def _read_repo_file(repo_root: str, rel_path: str) -> Optional[Tuple[str, str]]:
    rel_path = (rel_path or "").strip()
    if not rel_path:
        return None
    if os.path.isabs(rel_path):
        abs_path = rel_path
    else:
        abs_path = os.path.join(repo_root, rel_path)

    if (not os.path.isfile(abs_path)) and rel_path.startswith("../repos/"):
        parts = rel_path.replace("\\", "/").split("/")
        if len(parts) >= 4:
            fallback_rel = "/".join(parts[3:])
            fallback_abs = os.path.join(repo_root, fallback_rel)
            if os.path.isfile(fallback_abs):
                abs_path = fallback_abs
                rel_path = fallback_rel

    if not os.path.isfile(abs_path):
        return None
    try:
        content = open(abs_path, "r", encoding="utf-8", errors="ignore").read()
        rel_show = os.path.relpath(abs_path, repo_root)
        if rel_show.startswith(".."):
            rel_show = rel_path
        return (rel_show, content)
    except Exception:
        return None

def _normalize_path_lines(path_txt: Path) -> List[str]:
    try:
        lines = path_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    out = []
    for s in lines:
        s = (s or "").strip()
        if s:
            out.append(s)
    return out

def _ensure_codefile_for_instance(item: Dict[str, Any], results_dir: str) -> Dict[str, Any]:
    instance_id = item.get("instance_id")
    repo = item.get("repo")
    base_commit = item.get("base_commit")

    inst_dir = Path(results_dir) / instance_id
    req_path = inst_dir / "requirement.txt"
    issue_text = req_path.read_text(encoding="utf-8", errors="ignore")
    query = issue_text.strip() or ""

    path_txt = inst_dir / "path.txt"
    codefile_txt = inst_dir / "codefile.txt"

    if codefile_txt.exists():
        return {"instance_id": instance_id, "action": "skip", "reason": "codefile_exists"}

    repo_root = get_local_repo_path(repo)
    checkout_commit(repo_root, base_commit)

    if not path_txt.exists():
        _ = write_pathtext_from_requirement(inst_dir)
        pathtext_file = inst_dir / "pathtext.txt"
        path_query = pathtext_file.read_text(encoding="utf-8", errors="ignore") if pathtext_file.exists() else ""
        cand_rel_paths = candidate_paths_from_requirement(path_query)

        resolved_rel_paths = []
        seen = set()
        for rp in cand_rel_paths:
            rp = rp.strip().lstrip("/").replace("\\", "/")
            if not rp:
                continue
            abs_p = os.path.join(repo_root, rp)
            if os.path.isfile(abs_p) and rp not in seen:
                seen.add(rp)
                resolved_rel_paths.append(rp)

        if resolved_rel_paths:
            rel_paths = resolved_rel_paths
        else:
            selected = sparse_select_files(repo_root, query)
            rel_paths = [p for p, _ in selected]

        inst_dir.mkdir(parents=True, exist_ok=True)
        path_txt.write_text("\n".join(rel_paths) + ("\n" if rel_paths else ""), encoding="utf-8")

    rel_paths = _normalize_path_lines(path_txt)
    selected: List[Tuple[str, str]] = []
    for rel in rel_paths:
        got = _read_repo_file(repo_root, rel)
        if got:
            selected.append(got)

    if not selected:
        selected = sparse_select_files(repo_root, query)

    code_block = build_code_block(selected)
    inst_dir.mkdir(parents=True, exist_ok=True)
    codefile_txt.write_text(code_block, encoding="utf-8")
    return {"instance_id": instance_id, "action": "generated_codefile"}

def _all_codefiles_exist(items: List[Dict[str, Any]], results_dir: str) -> bool:
    for it in items:
        iid = it.get("instance_id")
        if not iid:
            continue
        if not (Path(results_dir) / iid / "codefile.txt").exists():
            return False
    return True

def _call_model_for_instance(item: Dict[str, Any], results_dir: str, out_dir: str, model: str, dataset: str) -> Dict[str, Any]:
    instance_id = item.get("instance_id")
    inst_dir = Path(results_dir) / instance_id
    req_path = inst_dir / "requirement.txt"
    issue_text = req_path.read_text(encoding="utf-8", errors="ignore")
    if (dataset or "") == "Pro":
        raw_interface = item.get("interface")
        interface = raw_interface.strip() if isinstance(raw_interface, str) else ""
    else:
        interface = ""

    path_txt = inst_dir / "path.txt"
    codefile_txt = inst_dir / "codefile.txt"

    raw_code_block = codefile_txt.read_text(encoding="utf-8", errors="ignore")
    prompt = build_prompt(issue_text, raw_code_block, interface)

    inst_out_dir = Path(out_dir) / instance_id
    inst_out_dir.mkdir(parents=True, exist_ok=True)
    (inst_out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    retrieval_method = "oracle" if path_txt.exists() else "bm25"
    print(f"{instance_id}: {retrieval_method}")

    if model == "qwen":
        model_resp = call_qwen(prompt)
    elif model == "deepseek":
        model_resp = call_deepseek(prompt)
    else:
        raise ValueError(f"Unsupported model: {model}")

    output_split = re.split(r"```requirement", model_resp)
    output_content = output_split[0].strip() if output_split else model_resp.strip()
    (inst_out_dir / "output.txt").write_text(output_content, encoding="utf-8")

    match = re.search(r"```patch\s*(.*?)\s*```", model_resp, flags=re.DOTALL)
    if match:
        model_resp_clean = match.group(1).strip()
    else:
        match_start = re.search(r"```patch\s*", model_resp, flags=re.DOTALL)
        if match_start:
            start_pos = match_start.end()
            model_resp_clean = model_resp[start_pos:].strip()
        else:
            model_resp_clean = model_resp.strip()

    out_patch_path = inst_out_dir / f"patch.patch"
    fixed_patch = fix_patch_text(model_resp_clean)
    out_patch_path.write_text(fixed_patch, encoding="utf-8")

    return {"instance_id": instance_id, "status": "model_called"}


def collect_patch_entries(out_dir: str) -> List[dict]:
    entries: List[dict] = []
    root = Path(out_dir)
    if not root.is_dir():
        return entries

    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        instance_id = entry.name
        patch_file = entry / "patch.patch"
        patch_content = ""
        if patch_file.is_file():
            try:
                patch_content = patch_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                patch_content = ""
        entries.append({"instance_id": instance_id, "patch": patch_content})
    return entries


def write_patch_summary_json(out_dir: str, file_name: str = "patches.json"):
    entries = collect_patch_entries(out_dir)
    out_path = file_name
    with open(out_path, "w", encoding="utf-8") as fo:
        json.dump(entries, fo, ensure_ascii=False, indent=2)
    print(f"[INFO] 已写入 {len(entries)} 条到 {out_path}")

def process_instances(
    input_jsonl: str,
    results_dir: str,
    out_dir: str,
    concurrency,
    dataset,
    model,
    summary_json_name,
):
    configure_dataset_rules(dataset)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as inf:
        items = [json.loads(line) for line in inf.read().splitlines() if line.strip()]

    round_idx = 0
    while not _all_codefiles_exist(items, results_dir):
        round_idx += 1
        made_progress = False
        for it in tqdm(items, desc=f"fill_codefile(round={round_idx})"):
            res = _ensure_codefile_for_instance(it, results_dir)
            if res["action"] == "generated_codefile":
                made_progress = True
        if not made_progress:
            break

    if not _all_codefiles_exist(items, results_dir):
        items = [
            it for it in items
            if (Path(results_dir) / it.get("instance_id", "") / "codefile.txt").exists()
        ]

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(_call_model_for_instance, it, results_dir, out_dir, model, dataset)
            for it in items
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"model_calls(conc={concurrency})"):
            fut.result()

    write_patch_summary_json(out_dir, summary_json_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ite", type=str)
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "deepseek"])
    parser.add_argument("--dataset", type=str, default="Pro", choices=["Lite", "Verified", "Pro"])
    args = parser.parse_args()
    input_file = f"../Datasets/{args.dataset}.jsonl"
    if args.ite == "0":
        results_dir = f"../Results/requirement/{args.model}/{args.dataset}/Compared/Requirement{args.ite}"
        out_dir = f"../Results/patch/{args.model}/{args.dataset}/Compared/patch{args.ite}"
        json_file = f"../Results/patch/{args.model}/{args.dataset}/Compared/patch{args.ite}/patch.json"
    else:
        results_dir = f"../Results/requirement/{args.model}/{args.dataset}/Generated/Requirement{args.ite}"
        out_dir = f"../Results/patch/{args.model}/{args.dataset}/Generated/patch{args.ite}"
        json_file = f"../Results/patch/{args.model}/{args.dataset}/Generated/patch{args.ite}/patch.json"

    process_instances(
        input_file,
        results_dir=results_dir,
        out_dir=out_dir,
        concurrency=20,
        dataset=args.dataset,
        model=args.model,
        summary_json_name=json_file,
    )

if __name__ == "__main__":
    main()
