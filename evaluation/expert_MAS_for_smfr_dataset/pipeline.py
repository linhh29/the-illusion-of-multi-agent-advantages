"""
Meta-agent orchestration pipeline.

Flow for each problem:
  Phase 1  (meta)         — MetaAgent parses the problem (1 LLM call)
  Phase 2  (transactions) — ExtractAgent gets each investor's transactions in parallel
  Phase 3  (prices)       — ExtractAgent looks up each transaction price in parallel
  Phase 4  (targets)      — CalculateAgent computes P&L + required sell price per investor
  Phase 5  (valid_dates)  — ExtractAgent finds valid sell dates per open position in parallel
  Phase 6  (aggregate)    — Pure Python: build final structured answer dict

All LLM calls return (response, CallUsage); usage is accumulated into a PipelineStats
object so run_pipeline() can return full cost/token data alongside the answer.
"""

import json
import asyncio
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents import (
    MetaAgent, ExtractAgent, CalculateAgent, ModelClient,
    ProblemMetaResponse, PipelineStats, CallUsage,
)

logger = logging.getLogger(__name__)

DATE_FMT = "%B %d, %Y"


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = re.sub(r'\b0(\d)\b', r'\1', s.strip())
    for fmt in ("%B %d, %Y", "%B %-d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _fmt_date(dt: datetime) -> str:
    return dt.strftime("%B %-d, %Y")


def _sort_dates(dates: List[str]) -> List[str]:
    parsed = [_parse_date(d) for d in dates]
    unique = sorted({dt for dt in parsed if dt})
    return [_fmt_date(dt) for dt in unique]


def _safe_json(s: str) -> Any:
    if s is None:
        return None
    s = s.strip()
    # Strip markdown code fences
    if s.startswith("```"):
        s = re.sub(r"^```[^\n]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s.strip())
    # Try direct parse
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    # Strip thousands-separator commas from numbers (e.g. 122,964.72 → 122964.72)
    s_clean = re.sub(r'(\d),(\d{3})', r'\1\2', s)
    s_clean = re.sub(r'(\d),(\d{3})', r'\1\2', s_clean)  # second pass for 1,234,567
    try:
        return json.loads(s_clean)
    except (json.JSONDecodeError, TypeError):
        pass
    # LLM wrapped JSON in prose — extract first {...} or [...]
    for pattern in (r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'):
        m = re.search(pattern, s_clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, TypeError):
                pass
    return s


# ---------------------------------------------------------------------------
# Phase 2: extract transactions with prices (merged phases 2+3)
# ---------------------------------------------------------------------------

async def _extract_transactions_with_prices(
    problem_text: str,
    investor: str,
    price_type: str,
    extract_agent: ExtractAgent,
    stats: PipelineStats,
) -> List[Dict]:
    """Extract all transactions for one investor AND look up each transaction's
    price from the smfr price history in a single LLM call."""
    query = (
        f'The context has two sections: (1) smfr price history tables '
        f'(columns: Date/Open/High/Low/Close/Volume), and (2) a transaction events '
        f'section with plain sentences like "Alice bought 54 shares of Apple."\n'
        f'Do the following in one pass for investor {investor}:\n'
        f'1. From the transaction events section ONLY, extract every transaction '
        f'(some companies will have only a buy or only a sell — include all of them).\n'
        f'2. For each transaction, look up the exact {price_type} price for that '
        f'company on that date from the smfr price history tables.\n'
        f'For action: normalize to exactly "buy" (bought/acquired/purchased) '
        f'or "sell" (sold/disposed/divested).\n'
        f'Return as a JSON array, e.g.\n'
        f'[{{"company": "Coca-Cola", "date": "December 29, 2025", "action": "buy", "quantity": 94, "price": 154.3600006104}}]'
    )
    resp, usage = await extract_agent.extract(problem_text, query)
    stats.add(usage, phase="transactions_prices")
    if resp is None:
        logger.warning("extract_transactions_with_prices returned None for %s", investor)
        return []
    result = _safe_json(resp.result)
    if isinstance(result, list):
        return result
    logger.warning("Unexpected transaction format for %s: %s", investor, resp.result)
    return []


# ---------------------------------------------------------------------------
# Phase 4: P&L summary + required price for one open position (single call)
# ---------------------------------------------------------------------------

async def _calculate_pnl_and_target(
    investor: str,
    transactions_with_prices: List[Dict],
    open_position: Dict,
    target_pct: float,
    question_type: str,
    calc_agent: CalculateAgent,
    stats: PipelineStats,
) -> Optional[Tuple[Dict, str, float, str]]:
    """Single LLM call: compute P&L + required price for the open position."""
    lines = [f"Investor: {investor}", "Transactions:"]
    for tx in transactions_with_prices:
        price_str = f"{float(tx['price']):.4f}" if tx.get("price") is not None else "UNKNOWN"
        lines.append(
            f"  {tx['action'].upper()} {tx['quantity']} shares of {tx['company']} "
            f"on {tx['date']} at price {price_str}"
        )
    data = "\n".join(lines)

    company = open_position["company"]
    open_price = float(open_position.get("price") or 0.0)
    quantity = float(open_position["quantity"])
    comparator = ">=" if question_type == "sell" else "<="

    if question_type == "sell":
        # Open position is a BUY. Find the required sell price.
        # required_sell_price = buy_price + profit_needed / quantity
        step5 = (
            f"5. required_price for the open position in {company} "
            f"(bought {quantity} shares at {open_price:.4f}):\n"
            f"   required_price = {open_price:.4f} + profit_needed / {quantity}\n"
            f"   (investor must sell at price >= required_price)"
        )
    else:
        # Open position is a SELL. Find the required buy price.
        # The buy cost B is unknown, so total cost = portfolio_cost + B.
        # We want: (realized_pnl + sell_revenue - B) / (portfolio_cost + B) = target_pct/100
        # Solving: B = (realized_pnl + sell_revenue - target_pct/100 × portfolio_cost) / (1 + target_pct/100)
        # required_price = B / quantity
        sell_revenue = open_price * quantity
        step5 = (
            f"5. required_price for the open position in {company} "
            f"(sold {quantity} shares at {open_price:.4f}, sell_revenue = {sell_revenue:.4f}):\n"
            f"   buy_cost = (realized_pnl + {sell_revenue:.4f} - {target_pct}/100 × total_cost) / (1 + {target_pct}/100)\n"
            f"   required_price = buy_cost / {quantity}\n"
            f"   (investor must buy at price <= required_price)"
        )

    query = (
        f"Calculate for investor {investor}:\n"
        f"1. total_cost = sum of (buy_price × quantity) for BUY transactions only\n"
        f"2. realized_pnl = sum of (sell_price − buy_price) × quantity for each completed pair\n"
        f"   A completed pair is a company that appears in BOTH a buy and a sell transaction.\n"
        f"3. target_profit = {target_pct}% × total_cost / 100\n"
        f"4. profit_needed = target_profit − realized_pnl  (can be negative if target already exceeded)\n"
        f"{step5}\n"
        f"Return as JSON: "
        f'{{"total_cost": float, "realized_pnl": float, "target_profit": float, '
        f'"profit_needed": float, "required_price": float}}'
    )

    resp, usage = await calc_agent.calculate(data, query)
    stats.add(usage, phase="pnl_and_target")
    if resp is None:
        return None
    result = _safe_json(resp.result)
    if not isinstance(result, dict):
        logger.warning("Unexpected pnl_and_target format for %s: %s", investor, resp.result)
        return None
    req_price = float(result.get("required_price", open_price))
    return result, company, req_price, comparator


# ---------------------------------------------------------------------------
# Phase 5: valid dates scan
# ---------------------------------------------------------------------------

async def _extract_valid_dates(
    problem_text: str,
    company: str,
    target_price: float,
    comparator: str,
    price_type: str,
    position_date: str,
    question_type: str,
    extract_agent: ExtractAgent,
    stats: PipelineStats,
) -> List[str]:
    if question_type == "sell":
        date_filter = f"Only include dates STRICTLY AFTER {position_date} (the buy date)."
    else:
        date_filter = f"Only include dates STRICTLY BEFORE {position_date} (the sell date)."

    query = (
        f"Scan the {company} smfr price history in the context and list ALL dates "
        f"where the {price_type} price is {comparator} {target_price:.4f}. "
        f"{date_filter} "
        f"Include every matching date — do not stop early. "
        f'Return as a JSON array in "Month Day, Year" format with no leading zeros on the day, '
        f'e.g. ["December 30, 2025", "January 5, 2026"]. '
        f"If no dates match, return []."
    )
    resp, usage = await extract_agent.extract(problem_text, query)
    stats.add(usage, phase="valid_dates")
    if resp is None:
        return []
    result = _safe_json(resp.result)
    if isinstance(result, list):
        return [str(d) for d in result if d]
    return []


# ---------------------------------------------------------------------------
# Phase 6: aggregate
# ---------------------------------------------------------------------------

def _build_final_answer(
    investors: List[str],
    inv_valid_dates: Dict[str, List[str]],
    aggregation: str,
) -> Dict:
    investor_dates = {}
    comparison = {}
    for inv in investors:
        sorted_dates = _sort_dates(inv_valid_dates.get(inv, []))
        investor_dates[inv] = sorted_dates
        comparison[inv] = sorted_dates[0] if sorted_dates else None

    candidates = {inv: _parse_date(d) for inv, d in comparison.items() if d}
    if not candidates:
        answer = None
    else:
        winning_dt = min(candidates.values()) if aggregation == "earliest" else max(candidates.values())
        winners = [inv for inv, dt in candidates.items() if dt == winning_dt]
        answer = winners

    return {"investor_dates": investor_dates, "comparison": comparison, "answer": answer}


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

async def run_pipeline(
    problem_text: str, model_client: ModelClient
) -> Tuple[Dict, PipelineStats, Dict]:
    """
    Run the full MAS pipeline for one problem.

    Returns:
        (answer_dict, pipeline_stats, trace)
        answer_dict  — investor_dates, comparison, answer
        pipeline_stats — token counts and cost breakdown
        trace        — all intermediate sub-agent outputs for analysis/debugging
    """
    stats = PipelineStats(model_name=model_client.model_name)
    trace: Dict = {}
    meta_agent = MetaAgent(model_client)
    extract_agent = ExtractAgent(model_client)
    calc_agent = CalculateAgent(model_client)

    empty_answer = {"investor_dates": {}, "comparison": {}, "answer": None}

    # ---- Phase 1: parse problem ----
    meta, usage = await meta_agent.parse(problem_text)
    stats.add(usage, phase="meta")
    if meta is None:
        logger.error("MetaAgent failed")
        return empty_answer, stats, trace

    trace["meta"] = meta.model_dump()
    logger.info(
        "Meta: investors=%s type=%s target=%.2f%% agg=%s price=%s",
        meta.investors, meta.question_type, meta.target_percentage,
        meta.aggregation, meta.price_type,
    )

    # ---- Phase 2: extract transactions + prices per investor (parallel) ----
    tx_results = await asyncio.gather(*[
        _extract_transactions_with_prices(problem_text, inv, meta.price_type, extract_agent, stats)
        for inv in meta.investors
    ])
    inv_transactions: Dict[str, List[Dict]] = dict(zip(meta.investors, tx_results))
    trace["transactions"] = inv_transactions

    # ---- Phases 4 + 5 per investor (parallel across investors) ----
    inv_trace: Dict[str, Dict] = {inv: {} for inv in meta.investors}

    async def _process_investor(inv: str) -> Tuple[str, List[str]]:
        txs = inv_transactions[inv]
        bought_companies = {tx["company"] for tx in txs if tx["action"] == "buy"}
        sold_companies   = {tx["company"] for tx in txs if tx["action"] == "sell"}

        if meta.question_type == "sell":
            open_positions = [tx for tx in txs if tx["action"] == "buy" and tx["company"] not in sold_companies]
        else:
            open_positions = [tx for tx in txs if tx["action"] == "sell" and tx["company"] not in bought_companies]

        inv_trace[inv]["open_positions"] = open_positions

        if not open_positions:
            logger.warning("No open positions for %s", inv)
            return inv, []

        async def _process_open_pos(pos: Dict) -> Tuple[str, Optional[float], str, List[str]]:
            result = await _calculate_pnl_and_target(
                inv, txs, pos, meta.target_percentage, meta.question_type, calc_agent, stats
            )
            if result is None:
                return pos["company"], None, "", []
            pnl_summary, company, req_price, comparator = result
            inv_trace[inv].setdefault("pnl", {})[company] = pnl_summary
            dates = await _extract_valid_dates(
                problem_text, company, req_price, comparator, meta.price_type,
                pos["date"], meta.question_type, extract_agent, stats
            )
            return company, req_price, comparator, dates

        pos_results = await asyncio.gather(*[_process_open_pos(pos) for pos in open_positions])

        inv_trace[inv]["targets"] = [
            {"company": company, "required_price": req_price, "comparator": comparator, "valid_dates": dates}
            for company, req_price, comparator, dates in pos_results
        ]

        all_dates: List[str] = []
        for _, _, _, dates in pos_results:
            all_dates.extend(dates)
        return inv, all_dates

    investor_results = await asyncio.gather(*[_process_investor(inv) for inv in meta.investors])
    inv_valid_dates: Dict[str, List[str]] = dict(investor_results)
    trace["investors"] = inv_trace

    # ---- Phase 6: aggregate ----
    answer = _build_final_answer(meta.investors, inv_valid_dates, meta.aggregation)
    return answer, stats, trace
