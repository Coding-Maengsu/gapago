"""
Pipeline stage-by-stage timing measurement.
Uses Gemini provider to avoid Azure dependency.
"""
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["LLM_PROVIDER"] = "gemini"

from graphs.graph import build_graph
from langchain_core.messages import HumanMessage

def main():
    query = "What are the current limitations of transformer-based models in low-resource NLP?"

    print(f"=== Pipeline Timing Test ===")
    print(f"Query: {query}")
    print(f"Provider: gemini (gemini-2.0-flash)")
    print(f"{'='*60}\n")

    app = build_graph()

    inputs = {
        "messages": [HumanMessage(content=query)],
        "llm_provider": "gemini",
        "output_language": "en",
        "session_id": "timing_test",
    }

    config = {
        "configurable": {"thread_id": "timing_test_001"},
        "recursion_limit": 30,
    }

    node_times = {}
    current_node = None
    node_start = None
    total_start = time.time()

    try:
        for event in app.stream(inputs, config, stream_mode="updates"):
            for node_name, node_output in event.items():
                now = time.time()

                # Close previous node timing
                if current_node and node_start:
                    elapsed = now - node_start
                    node_times[current_node] = elapsed
                    print(f"  ✓ {current_node}: {elapsed:.1f}s")

                # Start new node timing
                current_node = node_name
                node_start = now
                print(f"  → Starting: {node_name}")

        # Close last node
        if current_node and node_start:
            elapsed = time.time() - node_start
            node_times[current_node] = elapsed
            print(f"  ✓ {current_node}: {elapsed:.1f}s")

    except Exception as e:
        if current_node and node_start:
            elapsed = time.time() - node_start
            node_times[current_node] = elapsed
            print(f"  ✗ {current_node}: {elapsed:.1f}s (FAILED: {e})")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    total = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"Stage-by-Stage Results:")
    print(f"{'='*60}")
    for node, t in node_times.items():
        pct = (t / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {node:<25} {t:>7.1f}s  ({pct:>5.1f}%)  {bar}")
    print(f"{'─'*60}")
    print(f"  {'TOTAL':<25} {total:>7.1f}s")


if __name__ == "__main__":
    main()
