import os
import json
import shutil
import argparse

def init_runtime(ite: str, model: str, dataset: str):
    global now, last, SELECT_JSONL, CANDIDATE_FILES, PATCH_DIRS, REQUIREMENT_DIRS
    global TOTAL_PATCH_DIR, TOTAL_REQUIREMENT_DIR

    now = str(ite)
    last = str(int(now) - 1)

    SELECT_JSONL = f"../Results/testresult/{model}/{dataset}/Compared/choose_with_score{now}.jsonl"

    CANDIDATE_FILES = [
        f"../Results/testresult/{model}/{dataset}/Compared/result{last}.jsonl",
        f"../Results/testresult/{model}/{dataset}/Generated/result{now}.jsonl",
    ]

    PATCH_DIRS = {
        CANDIDATE_FILES[0]: f"../Results/patch/{model}/{dataset}/Compared/patch{last}",
        CANDIDATE_FILES[1]: f"../Results/patch/{model}/{dataset}/Generated/patch{now}",
    }
    REQUIREMENT_DIRS = {
        CANDIDATE_FILES[0]: f"../Results/requirement/{model}/{dataset}/Compared/Requirement{last}",
        CANDIDATE_FILES[1]: f"../Results/requirement/{model}/{dataset}/Generated/Requirement{now}",
    }

    TOTAL_PATCH_DIR = f"../Results/patch/{model}/{dataset}/Compared/patch{now}"
    TOTAL_REQUIREMENT_DIR = f"../Results/requirement/{model}/{dataset}/Compared/Requirement{now}"

    os.makedirs(TOTAL_PATCH_DIR, exist_ok=True)
    os.makedirs(TOTAL_REQUIREMENT_DIR, exist_ok=True)


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _has_source(instance_id: str, file_name: str) -> bool:
    src_patch_dir = os.path.join(PATCH_DIRS[file_name], instance_id)
    src_req_dir = os.path.join(REQUIREMENT_DIRS[file_name], instance_id)
    return os.path.exists(src_patch_dir) and os.path.exists(src_req_dir)


def _resolve_candidate_key(file_field: str) -> str:
    """Resolve row['file'] to a key in PATCH_DIRS/REQUIREMENT_DIRS.

    Supports both full paths and bare basenames such as 'result1.jsonl'.
    """
    raw = str(file_field or "").strip()
    base = os.path.basename(raw)

    if raw in PATCH_DIRS:
        return raw
    if base in PATCH_DIRS:
        return base

    for key in PATCH_DIRS.keys():
        if raw and (key.endswith(raw) or os.path.basename(key) == base):
            return key

    raise KeyError(
        f"Unknown file field: {file_field}. Available keys: "
        f"{[os.path.basename(k) for k in PATCH_DIRS.keys()]}"
    )


def _copy_from(file_name: str, instance_id: str):
    key = _resolve_candidate_key(file_name)
    src_patch_dir = os.path.join(PATCH_DIRS[key], instance_id)
    src_req_dir = os.path.join(REQUIREMENT_DIRS[key], instance_id)

    dst_patch_dir = os.path.join(TOTAL_PATCH_DIR, instance_id)
    dst_req_dir = os.path.join(TOTAL_REQUIREMENT_DIR, instance_id)

    if os.path.exists(dst_patch_dir):
        shutil.rmtree(dst_patch_dir)
    shutil.copytree(src_patch_dir, dst_patch_dir)

    if os.path.exists(dst_req_dir):
        shutil.rmtree(dst_req_dir)
    shutil.copytree(src_req_dir, dst_req_dir)

    print(f"Copied [{os.path.basename(key)}] {instance_id}")


def _collect_patch_entries(root: str):
    entries = []
    if not os.path.isdir(root):
        return entries

    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        instance_id = entry.name

        patch_content = ""
        candidates = [
            os.path.join(entry.path, "patch.patch"),
            os.path.join(entry.path, f"{instance_id}.patch"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        patch_content = f.read()
                except OSError:
                    patch_content = ""
                break

        entries.append({"instance_id": instance_id, "patch": patch_content})

    return entries


def _write_patch_json(root: str):
    out_path = os.path.join(root, "patch.json")
    entries = _collect_patch_entries(root)
    with open(out_path, "w", encoding="utf-8") as fo:
        json.dump(entries, fo, ensure_ascii=False, indent=2)
    print(f"Wrote {len(entries)} entries to {out_path}")


def run_choose():
    if not os.path.exists(SELECT_JSONL):
        raise FileNotFoundError(f"SELECT_JSONL not found: {SELECT_JSONL}")

    print(f"Reading select jsonl: {SELECT_JSONL}")

    for row in _iter_jsonl(SELECT_JSONL):
        instance_id = str(row.get("instance_id", "")).strip()
        if not instance_id:
            continue

        file_field = row.get("file") or ""
        _copy_from(str(file_field).strip(), instance_id)

    _write_patch_json(TOTAL_PATCH_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ite", type=str)
    parser.add_argument("--model", type=str, default="deepseek", choices=["qwen", "deepseek"])
    parser.add_argument("--dataset", type=str, default="Pro", choices=["Lite", "Verified", "Pro"])
    args = parser.parse_args()

    init_runtime(args.ite, args.model, args.dataset)
    run_choose()


if __name__ == "__main__":
    main()