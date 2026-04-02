"""
ModelRouter — 에이전트별 최적 모델 자동 배정 (프리셋 기반)

사용 가능 provider: azure, claude, groq (웹 배포 기준)

프로파일:
  optimized : 단순→groq, 추론→groq, 핵심 추출/응답→claude
  quality   : 핵심 작업 claude + 추론 groq (최고 품질)
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
    "optimized": {
        # azure(GPT) 기본: 정확성 필수 작업 전부 (환각 방지)
        # query_analysis, query_refine, meaning_expand, limitation_eval,
        # recency_check, critic_score, gap_classify, limitation_verify → 기본 provider
        "orchestrator":        AgentModelConfig(provider="groq", tier="light"),
        # heavy: 핵심 추론/추출 → claude, groq
        "limitation_extract":  AgentModelConfig(provider="claude", tier="heavy"),
        "gap_reasoning":       AgentModelConfig(provider="groq", tier="heavy"),
        "response":            AgentModelConfig(provider="claude", tier="heavy"),
    },
    "quality": {
        # heavy: 핵심 작업 전부 claude
        "limitation_extract":  AgentModelConfig(provider="claude", tier="heavy"),
        "limitation_eval":     AgentModelConfig(provider="claude", tier="heavy"),
        "gap_reasoning":       AgentModelConfig(provider="groq", tier="heavy"),
        "response":            AgentModelConfig(provider="claude", tier="heavy"),
        "recency_check":       AgentModelConfig(provider="claude", tier="standard"),
        "limitation_verify":   AgentModelConfig(provider="claude", tier="standard"),
        # light: 나머지 → 기본 provider (azure)
    },
}

# 프로파일별 기본 provider
PROFILE_DEFAULT_PROVIDERS: dict[str, str] = {
    "optimized": "azure",
    "quality":   "azure",
}


class ModelRouter:
    def __init__(self, default_provider: str, profile: str = "optimized"):
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
        return cls(d.get("default_provider", "azure"), d.get("profile", "optimized"))
