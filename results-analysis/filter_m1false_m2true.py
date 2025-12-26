#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python filter_m1false_m2true.py --model1 raw_outputs_lstm_test.jsonl --model2 raw_outputs_roberta_softmax_test.jsonl --output lstm_wrong_roberta_correct.jsonl
python filter_m1false_m2true.py --model1 raw_outputs_lstm_test.jsonl --model2 raw_outputs_xgboost_test.jsonl --output lstm_wrong_xgboost_correct.jsonl
python filter_m1false_m2true.py --model1 raw_outputs_xgboost_test.jsonl --model2 raw_outputs_roberta_softmax_test.jsonl --output xgboost_wrong_roberta_correct.jsonl

"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple


def read_jsonl(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    rows: List[Tuple[int, Dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path} line {line_no}: invalid JSON ({e}). Skipped.", file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                print(f"[WARN] {path} line {line_no}: JSON is not an object. Skipped.", file=sys.stderr)
                continue
            rows.append((line_no, obj))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Write model1 rows where model1 correct=false and model2 correct=true (match by field)."
    )
    parser.add_argument("--model1", required=True, help="Path to model1 JSONL file")
    parser.add_argument("--model2", required=True, help="Path to model2 JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--match_field", default="text", help="Field to match datapoints (default: text)")
    args = parser.parse_args()

    p1 = Path(args.model1).expanduser().resolve()
    p2 = Path(args.model2).expanduser().resolve()
    outp = Path(args.output).expanduser().resolve()
    key = args.match_field

    if not p1.exists() or not p1.is_file():
        print(f"[ERROR] model1 file not found: {p1}", file=sys.stderr)
        sys.exit(1)
    if not p2.exists() or not p2.is_file():
        print(f"[ERROR] model2 file not found: {p2}", file=sys.stderr)
        sys.exit(1)

    rows1 = read_jsonl(p1)
    rows2 = read_jsonl(p2)

    # Index model2 by match_field -> list of correct values (or objects)
    idx2 = defaultdict(list)
    for _, obj2 in rows2:
        k = obj2.get(key)
        if k is not None:
            idx2[k].append(obj2.get("correct"))

    outp.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    missing_in_model2 = 0
    ambiguous_matches = 0

    with outp.open("w", encoding="utf-8") as out:
        for _, obj1 in rows1:
            k = obj1.get(key)
            if k is None:
                continue

            # model1 must be incorrect
            if obj1.get("correct") is not False:
                continue

            cands = idx2.get(k, [])
            if not cands:
                missing_in_model2 += 1
                continue

            if len(cands) > 1:
                ambiguous_matches += 1

            # Keep if ANY matching row in model2 is correct==True
            if any(c is True for c in cands):
                out.write(json.dumps(obj1, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[OK] Wrote {kept} rows (model1 format) to: {outp}")
    if missing_in_model2:
        print(f"[INFO] {missing_in_model2} model1 incorrect rows had no match in model2 by '{key}'.")
    if ambiguous_matches:
        print(f"[INFO] {ambiguous_matches} model1 rows matched multiple rows in model2 (duplicate '{key}').")


if __name__ == "__main__":
    main()
