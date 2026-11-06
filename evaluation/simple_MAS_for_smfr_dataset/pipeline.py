"""
Simple MAS pipeline.

Flow for each problem:
  Phase 1 (meta)        — MetaAgent parses parameters (1 LLM call)  [identical to Expert MAS]
  Phase 2 (retrieval)   — RetrievalAgent extracts all transaction data (1 LLM call)
  Phase 3 (calculation) — CalculationAgent does all remaining reasoning (1 LLM call):
                           open-position identification, P&L, required price,
                           valid date scanning, aggregation, winner selection.

Total: 3 LLM calls per problem regardless of number of investors.
No Python scaffolding: no set-difference open-position detection, no formula
embedding, no Pydantic schema enforcement, no Python aggregation.
"""

import json
import re
import logging
from typing import Any, Dict, Optional, Tuple

from agents import (
    MetaAgent,
    RetrievalAgent,
    CalculationAgent,
    SimpleMASClient,
    PipelineStats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robust JSON parser (mirrors Expert MAS _safe_json)
# ---------------------------------------------------------------------------

def _parse_json(s: Any) -> Any:
    if s is None:
        return None
    if not isinstance(s, str):
        return s
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[^\n]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s.strip())
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    s_clean = re.sub(r'(\d),(\d{3})', r'\1\2', s)
    s_clean = re.sub(r'(\d),(\d{3})', r'\1\2', s_clean)
    try:
        return json.loads(s_clean)
    except (json.JSONDecodeError, TypeError):
        pass
    for pattern in (
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',
    ):
        m = re.search(pattern, s_clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, TypeError):
                pass
    return s


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

async def run_pipeline(
    problem_text: str,
    model_client: SimpleMASClient,
) -> Tuple[Dict, PipelineStats, Dict]:
    """
    Run the Simple MAS pipeline for one problem.

    Returns:
        (answer_dict, pipeline_stats, trace)
        answer_dict  — {"investor_dates": {...}, "answer": [...] or None}
        pipeline_stats — token counts and cost breakdown
        trace        — intermediate outputs for debugging
    """
    stats = PipelineStats(model_name=model_client.model_name)
    trace: Dict = {}
    empty_answer: Dict = {"investor_dates": {}, "answer": None}

    # ── Phase 1: parse problem parameters (MetaAgent, unchanged from Expert MAS) ──
    meta_agent = MetaAgent(model_client)
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

    # ── Phase 2: extract all transaction data (RetrievalAgent) ────────────────
    retrieval_agent = RetrievalAgent(model_client)
    raw_text, usage = await retrieval_agent.extract(
        problem_text, meta.investors, meta.price_type
    )
    stats.add(usage, phase="retrieval")

    extracted = _parse_json(raw_text)
    if not isinstance(extracted, dict):
        logger.warning("RetrievalAgent returned non-dict: %r", raw_text)
        extracted = {}
    trace["extracted"] = extracted

    # ── Phase 3: all reasoning → final answer (CalculationAgent) ──────────────
    calc_agent = CalculationAgent(model_client)
    raw_text, usage = await calc_agent.calculate(
        problem_text,
        json.dumps(extracted, indent=2),
        meta,
    )
    stats.add(usage, phase="calculation")

    answer = _parse_json(raw_text)
    if not isinstance(answer, dict):
        logger.warning("CalculationAgent returned non-dict: %r", raw_text)
        answer = empty_answer
    else:
        # Normalise: ensure required keys exist
        if "investor_dates" not in answer:
            answer["investor_dates"] = {}
        if "answer" not in answer:
            answer["answer"] = None
    trace["answer"] = answer

    return answer, stats, trace
