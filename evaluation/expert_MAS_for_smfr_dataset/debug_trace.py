"""
Debug script: compare pipeline trace against ground truth CoT for each phase.

Usage:
  # First run inference with --test to get a fresh trace:
  python run_inference.py --input <data_dir>/smfr_test.jsonl --model gpt-4.1 --test --output debug_out.jsonl

  # Then inspect (optionally pass fixed dataset to use corrected ground truth):
  python debug_trace.py debug_out.jsonl
  python debug_trace.py debug_out.jsonl <data_dir>/smfr_test.jsonl
"""

import json
import sys
import re
from pathlib import Path


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(v, decimals=4):
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def print_price_table(problem_text: str, company: str, price_type: str, threshold: float, comparator: str, after_date: str = None, before_date: str = None):
    """Extract and print the price table for a company from the raw problem text."""
    import re
    # Find the section for this company
    pattern = rf"{re.escape(company)} Historical Smfr Price Data\n(.*?)(?=\n\S[^\n]*Historical Smfr Price Data|\Z)"
    m = re.search(pattern, problem_text, re.DOTALL)
    if not m:
        print(f"  [Could not find {company} price table in problem text]")
        return

    rows = m.group(1).strip().split("\n")
    print(f"\n  {company} price table ({price_type}, threshold {comparator} {threshold:.4f}"
          + (f", after {after_date}" if after_date else "")
          + (f", before {before_date}" if before_date else "") + "):")
    # Show first 2 raw rows so we can verify the format
    print(f"  [raw sample] {rows[0][:120]!r}" if rows else "  [empty section]")
    print(f"  {'Date':<25} {price_type:<14} {'meets?'}")
    print(f"  {'-'*50}")

    date_pat = re.compile(r"Date:\s*(\S+)")
    price_pat = re.compile(rf"{price_type}:\s*([\d.]+)")

    for row in rows:
        dm = date_pat.search(row)
        pm = price_pat.search(row)
        if not dm or not pm:
            continue
        raw_date = dm.group(1)
        price = float(pm.group(1))

        # Check date filter
        in_range = True
        if after_date:
            in_range = in_range and (raw_date > after_date)
        if before_date:
            in_range = in_range and (raw_date < before_date)

        meets = (price >= threshold) if comparator == ">=" else (price <= threshold)
        flag = "✓" if (meets and in_range) else ("✗(date)" if (meets and not in_range) else "")
        if flag:
            print(f"  {raw_date:<25} {price:<14.4f} {flag}")


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def debug_record(idx, rec) -> bool:
    """Returns True if the final answer is correct."""
    section(f"Sample {idx+1}")

    inp = rec["input"]
    trace = rec.get("trace", {})
    output = rec.get("output", {})

    gen = inp.get("generation_params", {})
    print(f"  question_type : {gen.get('question_type')}")
    print(f"  aggregation   : {gen.get('aggregation')}")
    print(f"  price_type    : {gen.get('price_type')}")
    print(f"  target_pct    : {gen.get('target_percentage')}%")

    # ── Phase 1: meta ──────────────────────────────────────────────
    print(f"\n[Phase 1] Meta parse")
    meta = trace.get("meta", {})
    if meta:
        print(f"  investors     : {meta.get('investors')}")
        print(f"  question_type : {meta.get('question_type')}  (expected: {gen.get('question_type','?').replace('reverse_target_','')})")
        print(f"  target_pct    : {meta.get('target_percentage')}  (expected: {gen.get('target_percentage')})")
        print(f"  aggregation   : {meta.get('aggregation')}  (expected: {gen.get('aggregation')})")
        print(f"  price_type    : {meta.get('price_type')}  (expected: {gen.get('price_type')})")
    else:
        print("  (no meta trace)")

    # ── Ground truth answer ────────────────────────────────────────
    gt = inp.get("answer", {})
    gt_investor_dates = gt.get("investor_dates", {}) if isinstance(gt, dict) else {}
    gt_comparison     = gt.get("comparison", {}) if isinstance(gt, dict) else {}
    gt_answer         = gt.get("answer") if isinstance(gt, dict) else gt

    # ── Per-investor trace ─────────────────────────────────────────
    inv_traces = trace.get("investors", {})
    transactions = trace.get("transactions", {})

    # ── Ground truth CoT (full text, per investor) ────────────────
    cot_text = inp.get("cot", "")
    if cot_text:
        print(f"\n[Ground Truth CoT]")
        # Split by investor sections (each starts with the investor name on its own line)
        inv_names = list(inv_traces.keys())
        for inv in inv_names:
            # Find the block starting with this investor's name
            m = re.search(rf"^{re.escape(inv)}\n(.*?)(?=\n(?:{'|'.join(re.escape(i) for i in inv_names)})\n|\Z)",
                          cot_text, re.DOTALL | re.MULTILINE)
            if m:
                print(f"\n  --- {inv} ---")
                for line in m.group(0).strip().split("\n"):
                    print(f"  {line}")

    for inv, itrace in inv_traces.items():
        print(f"\n[Investor: {inv}]")

        # Transactions extracted
        txs = transactions.get(inv, [])
        print(f"  Transactions extracted ({len(txs)}):")
        for tx in txs:
            print(f"    {tx.get('action','?').upper():4s}  {tx.get('company','?'):15s}  "
                  f"{tx.get('date','?'):20s}  price={fmt(tx.get('price'))}")

        # Open positions
        ops = itrace.get("open_positions", [])
        print(f"  Open positions: {[p.get('company') for p in ops]}")

        # P&L computed vs ground truth CoT
        pnl_map = itrace.get("pnl", {})
        cot_text = inp.get("cot", "")

        # Try to extract ground truth numbers from CoT
        gt_cost_m   = re.search(rf"{inv}.*?Portfolio cost.*?\$([\d,.]+)", cot_text, re.DOTALL)
        gt_profit_m = re.search(rf"{inv}.*?Portfolio profit from completed.*?\$([\d,.]+)", cot_text, re.DOTALL)
        gt_needed_m = re.search(rf"{inv}.*?Profit needed.*?\$([\d,.]+)", cot_text, re.DOTALL)
        gt_req_m    = re.search(rf"{inv}.*?Required (?:sell|buy) price.*?\$([\d,.]+)", cot_text, re.DOTALL)

        for company, pnl in pnl_map.items():
            print(f"\n  P&L for open position in {company}:")
            print(f"    total_cost      : {fmt(pnl.get('total_cost'))}  "
                  f"(GT≈{gt_cost_m.group(1) if gt_cost_m else '?'})")
            print(f"    realized_pnl    : {fmt(pnl.get('realized_pnl'))}  "
                  f"(GT≈{gt_profit_m.group(1) if gt_profit_m else '?'})")
            print(f"    target_profit   : {fmt(pnl.get('target_profit'))}")
            print(f"    profit_needed   : {fmt(pnl.get('profit_needed'))}  "
                  f"(GT≈{gt_needed_m.group(1) if gt_needed_m else '?'})")
            print(f"    required_price  : {fmt(pnl.get('required_price'))}  "
                  f"(GT≈{gt_req_m.group(1) if gt_req_m else '?'})")

        # Targets + dates
        targets = itrace.get("targets", [])
        for t in targets:
            company = t.get("company")
            req = t.get("required_price")
            comp = t.get("comparator")
            dates_actual = t.get("valid_dates", [])
            dates_expected = gt_investor_dates.get(inv, [])

            print(f"\n  Valid dates for {company} (price {comp} {fmt(req)}):")
            print(f"    Got      ({len(dates_actual):3d}): {dates_actual[:5]}{'...' if len(dates_actual)>5 else ''}")
            print(f"    Expected ({len(dates_expected):3d}): {dates_expected[:5]}{'...' if len(dates_expected)>5 else ''}")
            first_actual   = dates_actual[0] if dates_actual else None
            first_expected = dates_expected[0] if dates_expected else None
            def _norm(d):
                return re.sub(r'\b0(\d)\b', r'\1', d) if isinstance(d, str) else d
            match = "✓" if _norm(first_actual) == _norm(first_expected) else "✗"
            print(f"    First date: {first_actual}  (expected: {first_expected})  {match}")

            if first_actual != first_expected and req is not None:
                # Find the open position date for this company to use as date filter
                pos_date = None
                for pos in ops:
                    if pos.get("company") == company:
                        try:
                            from datetime import datetime as _dt
                            pos_date = _dt.strptime(pos["date"], "%B %d, %Y").strftime("%Y-%m-%dT")
                        except Exception:
                            pass
                        break
                price_type = meta.get("price_type", "Close")
                question_type = meta.get("question_type", "sell")
                after = pos_date if question_type == "sell" else None
                before = pos_date if question_type == "buy" else None
                print_price_table(inp.get("problem", ""), company, price_type, req, comp, after, before)

    # ── Final answer ───────────────────────────────────────────────
    print(f"\n[Final answer]")
    print(f"  Got      : {output}")
    print(f"  Expected : {gt}")

    # Use the same compare_answers logic for parity
    sys.path.insert(0, str(Path(__file__).parent.parent / "evaluate"))
    from evaluate_smfr_answer_code import compare_answers
    correct = compare_answers(gt, output)

    # Winner-only check (just the answer field, order-insensitive)
    def _ans_set(a):
        if isinstance(a, list): return set(a)
        if a is None: return set()
        return {a}
    exp_winner = gt.get("answer") if isinstance(gt, dict) else gt
    act_winner = output.get("answer") if isinstance(output, dict) else output
    winner_correct = _ans_set(exp_winner) == _ans_set(act_winner)

    status = "✓ CORRECT" if correct else ("✓ WINNER ONLY" if winner_correct else "✗ WRONG")
    print(f"  {status}")
    return correct, winner_correct


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "debug_out.jsonl"
    records = load_jsonl(path)
    print(f"Loaded {len(records)} records from {path}")

    # Optionally load a fixed dataset and patch ground truth answers by problem text
    if len(sys.argv) > 2:
        dataset_path = sys.argv[2]
        dataset_by_problem = {r["problem"]: r for r in load_jsonl(dataset_path) if "problem" in r}
        matched = 0
        for rec in records:
            prob = rec.get("input", {}).get("problem", "")
            ds_rec = dataset_by_problem.get(prob)
            if ds_rec:
                rec["input"]["answer"] = ds_rec["answer"]
                matched += 1
        print(f"Fixed dataset: {dataset_path} — patched {matched}/{len(records)} ground truth answers")

    results = [debug_record(i, rec) for i, rec in enumerate(records)]
    correct_count = sum(c for c, _ in results)
    winner_count  = sum(w for _, w in results)
    n = len(records)
    print(f"\n{'═'*60}")
    print(f"  FULL ACCURACY   : {correct_count}/{n} ({correct_count/n*100:.1f}%)")
    print(f"  WINNER ACCURACY : {winner_count}/{n} ({winner_count/n*100:.1f}%)")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
