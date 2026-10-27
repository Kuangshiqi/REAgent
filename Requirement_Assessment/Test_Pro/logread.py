
from math import log
import os
import json
import re
from pathlib import Path
from typing import List, Union, Optional, Dict

_TEST_DIR = Path(__file__).resolve().parent


def load_patch_map(patch_path: str) -> Dict[str, str]:
    """Load {instance_id: patch} from a JSON/JSONL file."""
    patch_map: Dict[str, str] = {}
    if not patch_path:
        raise ValueError("patch_path is required")
    if not os.path.exists(patch_path):
        raise FileNotFoundError(f"patch_path not found: {patch_path}")

    path = Path(patch_path)
    if not path.is_file():
        raise ValueError(f"patch_path must be a file, got: {patch_path}")

    rows: List[Dict[str, object]] = []
    if str(path).endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = [x for x in data if isinstance(x, dict)]
        else:
            raise ValueError("patch json must be a JSON array")
    elif str(path).endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    else:
        raise ValueError(f"patch_path must end with .json or .jsonl, got: {patch_path}")

    for obj in rows:
        iid = obj.get("instance_id")
        patch = obj.get("model_patch")
        if patch is None:
            patch = obj.get("patch")
        if patch is None:
            patch = obj.get("test_patch")
        if isinstance(patch, list):
            patch = "\n\n".join(str(x) for x in patch)
        if iid is not None and patch is not None:
            patch_map[str(iid)] = str(patch)

    if not patch_map:
        print(f"[WARN] Loaded 0 patches from: {patch_path}")
    else:
        print(f"[INFO] Loaded {len(patch_map)} patches from: {patch_path}")
    return patch_map


def process_instance_logs(
    instance_ids_path: str,
    logs_path: str,
    patch_path: str,
):
    """Read per-round logs under a user-provided logs root path."""
    base_root = Path(logs_path)

    instance_ids: List[str] = []
    with open(instance_ids_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            instance_ids.append(str(obj["instance_id"]))

    patch_map = load_patch_map(patch_path)

    results: dict[str, List[Union[bool, str]]] = {}

    resolved_kw = "Resolved and Regression"
    reproduced_kw = "Reproduced and Regression"

    def _resolve_round_dir(round_idx: int) -> Optional[Path]:
        # Allow custom path templates like /path/to/{i} and common round naming styles.
        if "{i}" in logs_path:
            templated_dir = Path(logs_path.format(i=round_idx))
            return templated_dir if templated_dir.is_dir() else None

        plain_round = base_root / f"test_{round_idx}"
        if plain_round.is_dir():
            return plain_round

        matched = sorted(base_root.glob(f"*test_{round_idx}"))
        if matched:
            return matched[0]

        return None

    for iid in instance_ids:
        results[iid] = []

        for i in range(10):
            round_dir = _resolve_round_dir(i)
            if round_dir is None:
                results[iid].append("Error_Patch")
                continue

            log_path = round_dir / iid / "workspace" / "test_stdout.log"

            if not log_path.is_file():
                results[iid].append("Error_Patch")
                continue

            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if resolved_kw in content:
                results[iid].append(True)

            elif reproduced_kw in content:
                results[iid].append(False)

            else:
                results[iid].append("Error_test")

    normalized_patch_path = Path(patch_path).as_posix()
    match = re.search(
        r"^(.*?/)?Results/patch/([^/]+)/([^/]+)/([^/]+)/patch(\d+)/patch\.json$",
        normalized_patch_path,
    )
    if not match:
        raise ValueError(
            "patch_path format must be Results/patch/<model>/<dataset>/<type>/patch<ite>/patch.json"
        )
    model_name = match.group(2)
    run_type = match.group(4)
    iteration = match.group(5)
    out = Path(
        f"../../Results/testresult/{model_name}/Pro/{run_type}/result{iteration}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as outf:
        for iid, res_list in results.items():
            total = len(res_list)
            true_count = sum(1 for x in res_list if x is True)
            avg_ratio = true_count / total if total > 0 else 0.0

            outf.write(
                json.dumps(
                    {
                        "instance_id": iid,
                        "patch": patch_map.get(iid, ""),
                        "avgpassratio": avg_ratio,
                        "results": res_list,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path",type=str,required=True)
    parser.add_argument("--patch_path",type=str,required=True)
    
    args = parser.parse_args()

    process_instance_logs(
        "../../Datasets/Pro.jsonl",
        logs_path=args.log_path,
        patch_path=args.patch_path,
    )
