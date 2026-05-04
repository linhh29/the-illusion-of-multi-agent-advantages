# -*- coding: utf-8 -*-
"""Parse STOCKS result .txt and print accuracy and API usage (same format as eval_gpqa)."""
import json
import sys
import os

def eval_stocks(result_file):
    """Evaluate STOCKS results from txt file."""
    if not os.path.isfile(result_file):
        print(f"Error: File not found: {result_file}")
        return
    with open(result_file, 'r') as f:
        lines = f.readlines()

    if len(lines) < 1:
        print("Error: Result file is empty or incomplete")
        return

    # First line: acc list and accuracy
    accs_str = lines[0].strip()
    try:
        parts = accs_str.split()
        accs = eval(parts[0])
        accuracy = float(parts[-1])
        print(f"Accuracy: {accuracy:.4f} ({sum(accs)}/{len(accs)})")
    except Exception:
        print("Error parsing accuracy")

    if len(lines) >= 2:
        resp_cnt_str = lines[1].strip()
        try:
            parts = resp_cnt_str.split()
            total_calls = int(parts[0])
            avg_calls = float(parts[1])
            print(f"Total API calls: {total_calls}")
            print(f"Average API calls per question: {avg_calls:.2f}")
        except Exception:
            print("Error parsing API call counts")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_stocks.py <result_file>")
        print("Example: python eval_stocks.py stocks_gpt-4o_..._run1/stocks_validate_43.txt")
        sys.exit(1)
    eval_stocks(sys.argv[1])
