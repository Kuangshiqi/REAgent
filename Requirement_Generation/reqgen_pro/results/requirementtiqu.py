#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

pat = re.compile(r"<requirement>(.*?)</requirement>", re.DOTALL)

def main():
    parser = argparse.ArgumentParser(description="Extract requirement text from instance JSON files.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    root = Path(args.root)

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

        out_path = inst_dir / "requirement.txt"
        out_path.write_text(found, encoding="utf-8")


if __name__ == "__main__":
    main()
