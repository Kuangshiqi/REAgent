import argparse
import json
import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from run_tests_avg import run_reproduction_tests, txt_file_contains_string


def load_predictions_json(path: str) -> List[Dict[str, Any]]:
    """
    Load model patches from a JSON file: a JSON array of objects.
    Each object must include instance_id. Patch text may be in "patch" (SWE-bench_Pro style)
    or "model_patch"; "patch" is used when "model_patch" is absent.
    """
    if not str(path).endswith(".json"):
        raise ValueError(
            "--predictions_path must be a .json file (JSON array of objects with "
            'instance_id and patch or model_patch)'
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Predictions JSON must be a JSON array")
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict) or "instance_id" not in item:
            raise ValueError('Each predictions entry must be an object with "instance_id"')
        row = dict(item)
        mp = row.get("model_patch")
        if mp is None:
            mp = row.get("patch", "")
        row["model_patch"] = mp if mp is not None else ""
        out.append(row)
    return out

execution_results = dict()

_TEST_DIR = Path(__file__).resolve().parent


def run_reproduction_for_each_instance(args, lines, run_id, test_jsonl):
    instance_ids = [line["instance_id"] for line in lines]
    patches = [line.get("model_patch", "") for line in lines]

    results = run_reproduction_tests(
        instance_ids,
        patches,
        args.num_workers,
        run_id,
        args.instance_ids,
        args.timeout,
        testing_patches=False,
        apply_model_patch=True,
        test_jsonl=test_jsonl,
        dataset_name=args.dataset,
    )
    return results


def _run_reproduction_tests(args):
    run_rounds_and_aggregate(args)


def run_rounds_and_aggregate(args):
    predictions_lines = load_predictions_json(args.predictions_path)

    with open(args.test_jsonl, "r") as f:
        evaluation_tests = [json.loads(line) for line in f]

    # 构建 instance_id -> patch 列表
    test_map: Dict[str, List[str]] = {}
    for obj in evaluation_tests:
        iid = obj["instance_id"]
        patches = obj.get("test_patch", [])
        if isinstance(patches, str):
            patches = [patches]
        test_map[iid] = patches

    max_rounds = max((len(v) for v in test_map.values()), default=0)
    per_instance_results: Dict[str, List[bool]] = {p["instance_id"]: [] for p in predictions_lines}

    for round_idx in range(max_rounds):
        round_entries = []
        for obj in evaluation_tests:
            iid = obj["instance_id"]
            patches = test_map.get(iid, [])
            if len(patches) > round_idx:
                new_obj = dict(obj)
                new_obj["test_patch"] = patches[round_idx]
                round_entries.append(new_obj)

        if not round_entries:
            continue

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=f"_round_{round_idx}.jsonl") as tf:
            temp_path = tf.name
            for e in round_entries:
                tf.write(json.dumps(e) + "\n")

        round_run_id = f"{args.run_id}_round{round_idx}"
        round_log_subdir = Path(args.run_id) / f"round{round_idx}"

        _ = run_reproduction_for_each_instance(args, predictions_lines, round_run_id, temp_path)

        src_log_dir = _TEST_DIR / "logs" / "run_evaluation" / round_run_id
        dst_log_dir = _TEST_DIR / "logs" / "run_evaluation" / round_log_subdir
        if src_log_dir.exists():
            dst_log_dir.parent.mkdir(parents=True, exist_ok=True)
            if dst_log_dir.exists():
                shutil.rmtree(dst_log_dir)
            shutil.move(str(src_log_dir), str(dst_log_dir))

        for pred in predictions_lines:
            iid = pred["instance_id"]
            patches = test_map.get(iid, [])
            if len(patches) > round_idx:
                log_path = (
                    _TEST_DIR
                    / "logs"
                    / "run_evaluation"
                    / round_log_subdir
                    / "test"
                    / iid
                    / "test_output.txt"
                )
                expected_output = "Resolved and Regression"

                if log_path.is_file():
                    val = txt_file_contains_string(str(log_path), expected_output)
                else:
                    val = False

                per_instance_results.setdefault(iid, []).append(bool(val))

        try:
            os.remove(temp_path)
        except OSError:
            pass

    pred_path = Path(args.predictions_path)
    report_path = str(pred_path.with_name(f"{pred_path.stem}_reproduction_report.jsonl"))
    with open(report_path, "w") as outf:
        for pred in predictions_lines:
            iid = pred["instance_id"]
            patch_results = per_instance_results.get(iid, [])
            total = len(patch_results)
            passed = sum(1 for x in patch_results if x)
            avg = passed / total if total > 0 else 0.0
            outf.write(
                json.dumps(
                    {
                        "instance_id": iid,
                        "avgpassratio": avg,
                        "patch_results": patch_results,
                    }
                )
                + "\n"
            )

    print(f"✅ Saved regression summary to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="JSON array of predictions: each object has instance_id and patch or model_patch",
    )
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for running tests in seconds")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--test_jsonl", type=str, required=True)

    args = parser.parse_args()
    _run_reproduction_tests(args)


if __name__ == "__main__":
    main()
