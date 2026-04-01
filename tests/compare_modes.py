"""
고정 파이프라인 vs 오케스트레이터 모드 비교 스크립트
사용법: python tests/compare_modes.py "연구 주제"
"""
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: F401
from langchain_core.messages import HumanMessage


async def run_pipeline(query: str, orchestrator_mode: bool):
    """한 모드로 파이프라인 실행, 노드별 실행 로그 + 최종 state 반환"""
    os.environ["GAPAGO_ORCHESTRATOR"] = "1" if orchestrator_mode else "0"
    mode_label = "orchestrator" if orchestrator_mode else "fixed"

    # build_graph 캐시 방지: 매번 새로 import
    import importlib
    import graphs.graph
    if hasattr(graphs.graph, '_graph_cache'):
        graphs.graph._graph_cache = None
    importlib.reload(graphs.graph)
    from graphs.graph import build_graph

    app = build_graph()
    config_dict = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 30,
    }

    inputs = {
        "messages": [HumanMessage(content=query)],
        "max_iterations": 3,
        "research_domain": "auto",
        "year_range": "auto",
    }

    print(f"\n{'='*70}")
    print(f"  MODE: {mode_label.upper()} (GAPAGO_ORCHESTRATOR={os.environ['GAPAGO_ORCHESTRATOR']})")
    print(f"  QUERY: {query}")
    print(f"{'='*70}")

    node_log = []
    t0 = time.time()

    async for event in app.astream(inputs, config_dict, subgraphs=True):
        path, update = event
        if path:
            for node, values in update.items():
                if node == "__interrupt__":
                    print(f"  [{mode_label}] *** INTERRUPT — 자동 종료 ***")
                    return {"mode": mode_label, "interrupted": True, "nodes": node_log, "elapsed": time.time() - t0}
            continue

        for node, values in update.items():
            if node == "__interrupt__":
                print(f"  [{mode_label}] *** INTERRUPT — 자동 종료 ***")
                return {"mode": mode_label, "interrupted": True, "nodes": node_log, "elapsed": time.time() - t0}
            if node.startswith("__"):
                continue

            elapsed = time.time() - t0
            info = {"node": node, "elapsed_sec": round(elapsed, 1)}

            if isinstance(values, dict):
                if values.get("papers"):
                    info["papers"] = len(values["papers"])
                if values.get("limitations"):
                    info["limitations"] = len(values["limitations"])
                if values.get("gaps"):
                    info["gaps"] = len(values["gaps"])
                if values.get("completed_stages"):
                    info["completed_stages"] = values["completed_stages"]

            node_log.append(info)
            print(f"  [{mode_label}] {elapsed:6.1f}s | {node}", end="")
            for k in ("papers", "limitations", "gaps"):
                if k in info:
                    print(f" | {k}={info[k]}", end="")
            print()

    total = time.time() - t0
    final_state = app.get_state(config_dict)
    values = final_state.values if final_state else {}

    return {
        "mode": mode_label,
        "interrupted": False,
        "nodes": node_log,
        "node_sequence": [n["node"] for n in node_log],
        "elapsed": round(total, 1),
        "papers_count": len(values.get("papers", [])),
        "limitations_count": len(values.get("limitations", [])),
        "gaps_count": len(values.get("gaps", [])),
        "completed_stages": values.get("completed_stages", []),
    }


def print_comparison(r_fixed, r_orch):
    print(f"\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'':20s} {'FIXED':>12s} {'ORCHESTRATOR':>14s}")
    print(f"  {'-'*46}")
    print(f"  {'Elapsed (sec)':20s} {r_fixed['elapsed']:>12.1f} {r_orch['elapsed']:>14.1f}")
    print(f"  {'Total nodes':20s} {len(r_fixed['nodes']):>12d} {len(r_orch['nodes']):>14d}")
    print(f"  {'Papers':20s} {r_fixed.get('papers_count','-'):>12} {r_orch.get('papers_count','-'):>14}")
    print(f"  {'Limitations':20s} {r_fixed.get('limitations_count','-'):>12} {r_orch.get('limitations_count','-'):>14}")
    print(f"  {'Gaps':20s} {r_fixed.get('gaps_count','-'):>12} {r_orch.get('gaps_count','-'):>14}")

    print(f"\n  Node execution order:")
    print(f"    FIXED:        {' → '.join(r_fixed.get('node_sequence', []))}")
    print(f"    ORCHESTRATOR: {' → '.join(r_orch.get('node_sequence', []))}")

    if r_orch.get("completed_stages"):
        print(f"\n  Orchestrator completed_stages: {r_orch['completed_stages']}")

    print(f"\n{'='*70}")


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "Domain adaptation techniques for LLM fine-tuning"

    # 1) 고정 파이프라인
    r_fixed = await run_pipeline(query, orchestrator_mode=False)

    # 2) 오케스트레이터 모드
    r_orch = await run_pipeline(query, orchestrator_mode=True)

    # 3) 비교
    print_comparison(r_fixed, r_orch)

    # 4) 결과 저장
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    fname = f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    (out / fname).write_text(
        json.dumps({"fixed": r_fixed, "orchestrator": r_orch}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  결과 저장 → outputs/{fname}")


if __name__ == "__main__":
    asyncio.run(main())
