"""
AutoRAGBench — Report Generator

Renders benchmark results as a formatted ASCII table and optional JSON
report to stdout / file.
"""

import json
from typing import List, Dict, Any
from dataclasses import asdict

from app.benchmark.runner import VariantSummary


def _fmt(val: float, decimals: int = 3) -> str:
    """Format float for table display."""
    return f"{val:.{decimals}f}"


def render_ascii_table(summaries: List[VariantSummary]) -> str:
    """Render a leaderboard-style ASCII comparison table."""
    headers = [
        "Variant",
        "Overall (95% CI)",
        "Context",
        "Faithful",
        "AnsRel",
        "Ground",
        "Latency(ms)",
        "Cost($)",
        "CacheHit%",
        "Errors%",
    ]

    import math
    rows = []
    for s in sorted(summaries, key=lambda x: x.avg_overall_score, reverse=True):
        # Use 95% Bootstrap CI for overall score
        overall_str = f"{_fmt(s.avg_overall_score)} [{_fmt(s.ci_lower_95, 2)}-{_fmt(s.ci_upper_95, 2)}]"
            
        rows.append([
            s.label,
            overall_str,
            _fmt(s.avg_context_relevance),
            _fmt(s.avg_faithfulness),
            _fmt(s.avg_answer_relevance),
            _fmt(s.avg_groundedness),
            _fmt(s.avg_latency_ms, 0),
            _fmt(s.total_token_cost, 6),
            _fmt(s.cache_hit_rate * 100, 1),
            _fmt(s.error_rate * 100, 1),
        ])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Build table
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

    lines = [
        "",
        "+-----------------------------------------------------------------------+",
        "|              AutoRAGBench -- Pipeline Variant Leaderboard             |",
        "+-----------------------------------------------------------------------+",
        "",
        sep,
        header_line,
        sep,
    ]
    for row in rows:
        row_line = "|" + "|".join(f" {cell:<{col_widths[i]}} " for i, cell in enumerate(row)) + "|"
        lines.append(row_line)
    lines.append(sep)
    lines.append("")

    # Add winner announcement
    if summaries:
        best = max(summaries, key=lambda s: s.avg_overall_score)
        cheapest = min(summaries, key=lambda s: s.total_token_cost)
        fastest = min(summaries, key=lambda s: s.avg_latency_ms)
        lines.append(f"[Best Quality]:  {best.label} (overall={_fmt(best.avg_overall_score)})")
        lines.append(f"[Cheapest]:      {cheapest.label} (cost=${_fmt(cheapest.total_token_cost, 6)})")
        lines.append(f"[Fastest]:       {fastest.label} (latency={_fmt(fastest.avg_latency_ms, 0)}ms)")
        lines.append("")

    # If any deterministic metrics exist, print a second table
    has_retrieval_metrics = any(s.avg_recall_at_5 > 0 or s.avg_mrr > 0 for s in summaries)
    if has_retrieval_metrics:
        ret_headers = ["Variant", "Recall@5", "Precision@5", "MRR", "nDCG@5"]
        ret_rows = []
        for s in sorted(summaries, key=lambda x: x.avg_ndcg_at_5, reverse=True):
            ret_rows.append([
                s.label,
                _fmt(s.avg_recall_at_5),
                _fmt(s.avg_precision_at_5),
                _fmt(s.avg_mrr),
                _fmt(s.avg_ndcg_at_5)
            ])
            
        ret_col_widths = [len(h) for h in ret_headers]
        for row in ret_rows:
            for i, cell in enumerate(row):
                ret_col_widths[i] = max(ret_col_widths[i], len(cell))
                
        ret_sep = "+" + "+".join("-" * (w + 2) for w in ret_col_widths) + "+"
        ret_header_line = "|" + "|".join(f" {h:<{ret_col_widths[i]}} " for i, h in enumerate(ret_headers)) + "|"
        
        lines.extend([
            "+-----------------------------------------------------------------------+",
            "|                Deterministic Retrieval Quality Metrics                |",
            "+-----------------------------------------------------------------------+",
            "",
            ret_sep,
            ret_header_line,
            ret_sep
        ])
        for row in ret_rows:
            lines.append("|" + "|".join(f" {cell:<{ret_col_widths[i]}} " for i, cell in enumerate(row)) + "|")
        lines.append(ret_sep)
        lines.append("")

    return "\n".join(lines)


def render_per_query_detail(summaries: List[VariantSummary]) -> str:
    """Render detailed per-query breakdown for each variant."""
    lines = []
    for s in summaries:
        lines.append(f"\n{'='*60}")
        lines.append(f"  {s.label} ({s.variant})")
        lines.append(f"  {s.description}")
        lines.append(f"{'='*60}")
        for qr in s.query_results:
            status = "✅" if qr.error is None else "❌"
            lines.append(
                f"  {status} [{qr.query_id}] "
                f"overall={_fmt(qr.ragas_overall)} "
                f"latency={_fmt(qr.latency_ms, 0)}ms "
                f"risk={qr.hallucination_risk}"
            )
            if qr.error:
                lines.append(f"     ERROR: {qr.error[:120]}")
    lines.append("")
    return "\n".join(lines)


def generate_json_report(summaries: List[VariantSummary]) -> Dict[str, Any]:
    """Generate a structured JSON report dict."""
    return {
        "benchmark": "AutoRAGBench",
        "version": "1.0",
        "total_variants": len(summaries),
        "total_queries_per_variant": summaries[0].total_queries if summaries else 0,
        "leaderboard": [
            {
                "rank": i + 1,
                "variant": s.variant,
                "label": s.label,
                "avg_overall_score": s.avg_overall_score,
                "avg_latency_ms": s.avg_latency_ms,
                "total_cost_usd": s.total_token_cost,
                "cache_hit_rate": s.cache_hit_rate,
                "error_rate": s.error_rate,
            }
            for i, s in enumerate(
                sorted(summaries, key=lambda x: x.avg_overall_score, reverse=True)
            )
        ],
        "variants": [asdict(s) for s in summaries],
    }


def save_report(summaries: List[VariantSummary], filepath: str) -> str:
    import json
    import os
    import subprocess
    from dataclasses import asdict
    
    commit_hash = "unknown"
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        pass

    data = {
        "metadata": {
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            "git_commit": commit_hash,
            "dataset_version": "v1.1_expanded",
            "environment": os.getenv("ENVIRONMENT", "local"),
        },
        "summaries": [asdict(s) for s in summaries]
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    
    return os.path.abspath(filepath)
