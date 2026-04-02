"""
ModelRouter — 에이전트별 최적 모델 자동 배정 (프리셋 기반)

프로파일:
  balanced  : 모든 에이전트가 기본 provider 사용 (현재 동작과 동일)
  optimized : 단순→gemini, 추론→groq, 핵심 추출/응답→claude
  quality   : 핵심 작업 claude + 추론 groq (최고 품질)
  speed     : 가능한 한 모든 에이전트를 경량 모델로 라우팅
"""

from dataclasses import dataclass, field
from llm import get_llm


@dataclass
class AgentModelConfig:
    provider: str | None = None   # None → 기본 provider
    model: str | None = None
    tier: str = "standard"        # "light" / "standard" / "heavy"


# 프리셋 정의
ROUTING_PRESETS: dict[str, dict[str, AgentModelConfig]] = {
    "balanced": {},  # 모든 에이전트 기본 provider (현재 동작)
    "optimized": {
        # light: 단순 분류/점수화 → gemini (빠르고 저렴)
        "query_analysis":      AgentModelConfig(provider="gemini", tier="light"),
        "query_refine":        AgentModelConfig(provider="gemini", tier="light"),
        "meaning_expand":      AgentModelConfig(provider="gemini", tier="light"),
        "critic_score":        AgentModelConfig(provider="gemini", tier="light"),
        "orchestrator":        AgentModelConfig(provider="gemini", tier="light"),
        "gap_classify":        AgentModelConfig(provider="gemini", tier="light"),
        # standard: limitation_eval, recency_check → 기본 provider
        # heavy: 핵심 추론/추출 → claude, groq
        "limitation_extract":  AgentModelConfig(provider="claude", tier="heavy"),
        "gap_reasoning":       AgentModelConfig(provider="groq", tier="heavy"),
        "response":            AgentModelConfig(provider="claude", tier="heavy"),
        "limitation_verify":   AgentModelConfig(provider="gemini", tier="light"),
    },
    "quality": {
        # light: 단순 작업만 gemini
        "orchestrator":        AgentModelConfig(provider="gemini", tier="light"),
        "gap_classify":        AgentModelConfig(provider="gemini", tier="light"),
        # heavy: 핵심 작업 전부 claude
        "limitation_extract":  AgentModelConfig(provider="claude", tier="heavy"),
        "limitation_eval":     AgentModelConfig(provider="claude", tier="heavy"),
        "gap_reasoning":       AgentModelConfig(provider="groq", tier="heavy"),
        "response":            AgentModelConfig(provider="claude", tier="heavy"),
        "recency_check":       AgentModelConfig(provider="claude", tier="standard"),
        "limitation_verify":   AgentModelConfig(provider="claude", tier="standard"),
    },
    "speed": {
        "query_analysis":      AgentModelConfig(provider="gemini", tier="light"),
        "query_refine":        AgentModelConfig(provider="gemini", tier="light"),
        "meaning_expand":      AgentModelConfig(provider="gemini", tier="light"),
        "limitation_eval":     AgentModelConfig(provider="gemini", tier="light"),
        "recency_check":       AgentModelConfig(provider="gemini", tier="light"),
        "critic_score":        AgentModelConfig(provider="gemini", tier="light"),
        "orchestrator":        AgentModelConfig(provider="gemini", tier="light"),
        "gap_classify":        AgentModelConfig(provider="gemini", tier="light"),
        "gap_reasoning":       AgentModelConfig(provider="groq", tier="heavy"),
        "response":            AgentModelConfig(provider="gemini", tier="light"),
        "limitation_verify":   AgentModelConfig(provider="gemini", tier="light"),
    },
}


class ModelRouter:
    def __init__(self, default_provider: str, profile: str = "balanced"):
        self.default_provider = default_provider
        self.profile = profile
        self.overrides = ROUTING_PRESETS.get(profile, {})

    def get_llm(self, agent_name: str):
        cfg = self.overrides.get(agent_name)
        if cfg and cfg.provider:
            return get_llm(provider=cfg.provider, model=cfg.model)
        return get_llm(provider=self.default_provider)

    def get_provider(self, agent_name: str) -> str:
        cfg = self.overrides.get(agent_name)
        if cfg and cfg.provider:
            return cfg.provider
        return self.default_provider

    def has_config(self, agent_name: str) -> bool:
        """해당 에이전트에 대한 라우팅 설정이 있는지 확인"""
        return agent_name in self.overrides

    def to_dict(self) -> dict:
        return {"default_provider": self.default_provider, "profile": self.profile}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelRouter":
        return cls(d.get("default_provider", "azure"), d.get("profile", "balanced"))
