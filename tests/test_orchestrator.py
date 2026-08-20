"""
오케스트레이터 단위 테스트
- orchestrator_node: 다양한 state 조합으로 올바른 다음 에이전트 결정 확인
- _wrap_agent: completed_stages 정상 누적, agent_feedback 정상 추출
- _dependencies_met: 필수 경로 제약 검증
- 최대 스텝 제한 및 무한 루프 방지
"""

import os
import sys
import pytest

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gapago.agents.orchestrator_agent import (
    _dependencies_met,
    _next_mandatory,
    _parse_decision,
    orchestrator_node,
    MANDATORY_SEQUENCE,
    MAX_ORCHESTRATOR_STEPS,
    MAX_AGENT_RERUNS,
)


class TestDependenciesMet:
    def test_meaning_expand_no_deps(self):
        assert _dependencies_met("meaning_expand", []) is True

    def test_paper_retrieval_needs_expand(self):
        assert _dependencies_met("paper_retrieval", []) is False
        assert _dependencies_met("paper_retrieval", ["meaning_expand"]) is True

    def test_limitation_extract_needs_retrieval(self):
        assert _dependencies_met("limitation_extract", ["meaning_expand"]) is False
        assert _dependencies_met("limitation_extract", ["meaning_expand", "paper_retrieval"]) is True

    def test_gap_infer_needs_limitation(self):
        assert _dependencies_met("gap_infer", ["meaning_expand", "paper_retrieval"]) is False
        assert _dependencies_met("gap_infer", ["meaning_expand", "paper_retrieval", "limitation_extract"]) is True

    def test_final_response_needs_gap(self):
        completed = ["meaning_expand", "paper_retrieval", "limitation_extract", "gap_infer"]
        assert _dependencies_met("final_response", completed) is True
        assert _dependencies_met("final_response", completed[:3]) is False


class TestNextMandatory:
    def test_empty_state(self):
        assert _next_mandatory([]) == "meaning_expand"

    def test_after_expand(self):
        assert _next_mandatory(["meaning_expand"]) == "paper_retrieval"

    def test_after_all_mandatory(self):
        completed = list(MANDATORY_SEQUENCE)
        assert _next_mandatory(completed) is None

    def test_with_optional_inserted(self):
        completed = ["meaning_expand", "paper_retrieval", "limitation_extract", "limitation_eval"]
        assert _next_mandatory(completed) == "gap_infer"


class TestParseDecision:
    def test_valid_json(self):
        resp = '{"next_agent": "meaning_expand", "reasoning": "first step"}'
        assert _parse_decision(resp, "fallback") == "meaning_expand"

    def test_end_signal(self):
        resp = '{"next_agent": "__end__", "reasoning": "all done"}'
        assert _parse_decision(resp, "fallback") == "__end__"

    def test_invalid_agent(self):
        resp = '{"next_agent": "nonexistent", "reasoning": "?"}'
        assert _parse_decision(resp, "fallback") == "fallback"

    def test_malformed_json(self):
        resp = "this is not json"
        assert _parse_decision(resp, "meaning_expand") == "meaning_expand"

    def test_json_in_code_block(self):
        resp = '```json\n{"next_agent": "gap_infer", "reasoning": "test"}\n```'
        assert _parse_decision(resp, "fallback") == "gap_infer"

    def test_optional_agent(self):
        resp = '{"next_agent": "critic_score", "reasoning": "quality gate"}'
        assert _parse_decision(resp, "fallback") == "critic_score"


class TestOrchestratorNode:
    def _base_state(self, **overrides):
        state = {
            "completed_stages": [],
            "agent_feedback": {},
            "orchestrator_plan": [],
            "messages": [],
            "papers": [],
            "limitations": [],
            "gaps": [],
            "llm_provider": "azure",
        }
        state.update(overrides)
        return state

    def test_max_steps_forces_final_response(self):
        """MAX_ORCHESTRATOR_STEPS 도달 시 final_response 강제"""
        completed = ["meaning_expand"] * MAX_ORCHESTRATOR_STEPS
        state = self._base_state(completed_stages=completed)
        cmd = orchestrator_node(state)
        assert cmd.goto == "final_response"

    def test_max_steps_with_final_done_goes_to_end(self):
        """final_response까지 완료된 상태에서 max steps → END"""
        completed = ["meaning_expand"] * (MAX_ORCHESTRATOR_STEPS - 1) + ["final_response"]
        state = self._base_state(completed_stages=completed)
        cmd = orchestrator_node(state)
        assert cmd.goto == "__end__"

    def test_all_mandatory_done_goes_to_end(self):
        """모든 필수 단계 완료 → END"""
        state = self._base_state(completed_stages=list(MANDATORY_SEQUENCE))
        cmd = orchestrator_node(state)
        assert cmd.goto == "__end__"


class TestExtractFeedback:
    def test_limitation_eval_retry(self):
        from gapago.graphs.orchestrator_graph import _extract_feedback
        result = {
            "limitation_eval": {"decision": "RETRY", "retry_guidance": "improve quality"},
        }
        fb = _extract_feedback("limitation_eval", result, {})
        assert fb["type"] == "quality_low"
        assert fb["from"] == "limitation_eval"

    def test_limitation_eval_pass(self):
        from gapago.graphs.orchestrator_graph import _extract_feedback
        result = {"limitation_eval": {"decision": "PASS"}}
        fb = _extract_feedback("limitation_eval", result, {})
        assert fb == {}

    def test_critic_redo_retrieval(self):
        from gapago.graphs.orchestrator_graph import _extract_feedback
        from langchain_core.messages import AIMessage
        result = {"messages": [AIMessage(content="DECISION: REDO_RETRIEVAL", name="critic")]}
        fb = _extract_feedback("critic_score", result, {})
        assert fb["type"] == "relevance_low"

    def test_unknown_agent_no_feedback(self):
        from gapago.graphs.orchestrator_graph import _extract_feedback
        fb = _extract_feedback("meaning_expand", {"messages": []}, {})
        assert fb == {}


class TestWrapAgent:
    def test_sync_agent_wrapper(self):
        from gapago.graphs.orchestrator_graph import _wrap_agent

        def mock_agent(state):
            return {"messages": [], "sender": "test"}

        wrapped = _wrap_agent(mock_agent, "test_agent")
        result = wrapped({})
        assert result["completed_stages"] == ["test_agent"]
        assert "agent_feedback" in result

    def test_async_agent_wrapper(self):
        import asyncio
        from gapago.graphs.orchestrator_graph import _wrap_agent

        async def mock_async_agent(state):
            return {"messages": [], "sender": "test"}

        wrapped = _wrap_agent(mock_async_agent, "async_agent")
        result = asyncio.get_event_loop().run_until_complete(wrapped({}))
        assert result["completed_stages"] == ["async_agent"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
