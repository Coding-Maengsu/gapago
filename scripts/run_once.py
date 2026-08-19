"""
비대화식 1회 실행 스크립트
사용법: python scripts/run_once.py "연구 주제"
"""
import asyncio
import json
import uuid
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401
from graphs.graph import build_graph
from langchain_core.messages import HumanMessage

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


async def run_once(query: str, provider: str = "azure", domain: str = "auto", year_range: str = "auto"):
    import os
    os.environ["LLM_PROVIDER"] = provider

    from llm import get_llm
    get_llm.cache_clear()
    print(f"[init] LLM provider: {provider}")
    get_llm(provider=provider)

    reasoning_provider = os.getenv("GAP_REASONING_PROVIDER", "")
    if reasoning_provider:
        get_llm(provider=reasoning_provider)

    app = build_graph()
    config_dict = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 30,
    }

    inputs = {
        "messages": [HumanMessage(content=query)],
        "max_iterations": 3,
        "research_domain": domain,
        "year_range": year_range,
    }

    print(f"[start] query: {query}")
    print(f"[start] domain: {domain}, year_range: {year_range}")
    print("=" * 70)

    # stream events
    async for event in app.astream(inputs, config_dict, subgraphs=True):
        path, update = event
        if path:
            for node, values in update.items():
                if node == "__interrupt__":
                    print("\n*** INTERRUPT — 대화형 입력 필요, 자동 종료 ***")
                    return
            continue

        for node, values in update.items():
            if node == "__interrupt__":
                print("\n*** INTERRUPT — 대화형 입력 필요, 자동 종료 ***")
                return
            print(f"\n--- {node} ---")
            if isinstance(values, dict):
                for key in ["iteration", "refined_query", "scope_level"]:
                    if key in values:
                        print(f"  {key} = {values[key]}")
                papers = values.get("papers")
                if papers:
                    print(f"  papers: {len(papers)}편")
                lims = values.get("limitations")
                if lims:
                    print(f"  limitations: {len(lims)}건")
                gaps = values.get("gaps")
                if gaps:
                    print(f"  gaps: {len(gaps)}건")
                status = values.get("paper_extraction_status")
                if status:
                    print(f"  paper_extraction_status: {len(status)}편 추적")

    # save result
    final_state = app.get_state(config_dict)
    values = final_state.values if final_state else {}

    messages_out = []
    for msg in values.get("messages", []):
        messages_out.append({
            "type": msg.type,
            "name": getattr(msg, "name", None),
            "content": getattr(msg, "content", ""),
        })

    result = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "refined_query": values.get("refined_query", ""),
        "keywords": values.get("keywords", []),
        "papers": [p.dict() if hasattr(p, 'dict') else p for p in values.get("papers", [])],
        "total_candidates_count": values.get("total_candidates_count", 0),
        "limitations": values.get("limitations", []),
        "paper_extraction_status": values.get("paper_extraction_status", []),
        "gaps": values.get("gaps", []),
        "web_results": values.get("web_results", []),
        "messages": messages_out,
    }

    fname = f"gapago_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = OUTPUT_DIR / fname
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'=' * 70}")
    print(f"결과 저장 완료 → {path}")

    # paper_extraction_status 요약
    status_list = values.get("paper_extraction_status", [])
    if status_list:
        print(f"\n=== paper_extraction_status 요약 ===")
        for s in status_list:
            print(f"  [{s['status']:17s}] {s['paper_id']:<20s} | source: {s['fulltext_source']:<10s} | lims: {s['limitations_count']} | sections: {s.get('sections_found', [])}")
    else:
        print("\npaper_extraction_status 없음 (코드에 반영 안 됨)")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Domain adaptation"
    asyncio.run(run_once(query))
