#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

pat = re.compile(r"<requirement>(.*?)</requirement>", re.DOTALL)

def main():
    parser = argparse.ArgumentParser(description="Extract requirement text from instance JSON files.")
    parser.add_argument("--ite", type=str)
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "deepseek"])
    parser.add_argument("--dataset", type=str, default="pro", choices=["Lite", "Verified", "Pro"])

    args = parser.parse_args()
    if args.dataset == "Pro":
        root = Path(f"./reqgen_pro/results/SWE-bench_SWE-bench_Pro_{args.model}_{args.ite}")
    else:
        root = Path(f"./reqgen_lv/results/SWE-bench_SWE-bench_{args.dataset}_{args.model}_{args.ite}")
    if args.ite == "0":
        save_root = Path(f"../Results/requirement/{args.model}/{args.dataset}/Compared/Requirement{args.ite}")
    else:
        save_root = Path(f"../Results/requirement/{args.model}/{args.dataset}/Generated/Requirement{args.ite}")
    save_root.mkdir(parents=True, exist_ok=True)

    for inst_dir in root.iterdir():
        if not inst_dir.is_dir():
            continue

        json_path = inst_dir / f"{inst_dir.name}.json"
        try:
            raw = json_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        matches = pat.findall(raw)

        if not matches:
            start_tag = "<requirement>"
            end_tag = "</requirement>"
            start_idx = raw.lower().rfind(start_tag)
            if start_idx != -1:
                tail = raw[start_idx + len(start_tag):]
                if end_tag not in tail.lower():
                    matches = [tail]

        found = None

        for m in reversed(matches):
            content = m.strip()

            try:
                content = bytes(content, "utf-8").decode("unicode_escape")
            except Exception:
                continue

            if not content:
                continue
            if content.strip() == "...":
                continue
            if content.strip() == "and":
                continue
            if content.strip() == "`and":
                continue
            if "Describe the main purpose or capabilities of the repository" in content:
                continue

            found = content
            break

        dst_inst_dir = save_root / inst_dir.name
        dst_inst_dir.mkdir(parents=True, exist_ok=True)
        out_path = dst_inst_dir / "requirement.txt"
        out_path.write_text(found or "", encoding="utf-8")


if __name__ == "__main__":
    main()
