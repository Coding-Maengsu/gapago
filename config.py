from dataclasses import dataclass, field
from typing import Optional
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv
from utils import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

load_dotenv(ENV_PATH, override=False)
logging.langsmith()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _str_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Configuration:
    tavily_max_results: int = field(
        default=_int_env("TAVILY_MAX_RESULTS", 5),
        metadata={
            "description": "Maximum number of Tavily search results",
            "range": [1, 50],
        },
    )
    arxiv_max_docs: int = field(
        default=_int_env("ARXIV_MAX_RESULTS", 20),
        metadata={"description": "Maximum number of ArXiv documents", "range": [1, 50]},
    )
    scienceon_client_id: Optional[str] = field(
        default=_str_env("SCIENCEON_CLIENT_ID"),
        metadata={"description": "ScienceON client ID"},
    )
    scienceon_mac_address: Optional[str] = field(
        default=_str_env("SCIENCEON_MAC_ADDRESS"),
        metadata={"description": "ScienceON registered MAC address"},
    )
    scienceon_key: Optional[str] = field(
        default=_str_env("SCIENCEON_KEY"),
        metadata={"description": "ScienceON AES key for token issuance"},
    )
    scienceon_default_target: str = field(
        default=_str_env("SCIENCEON_DEFAULT_TARGET", "ARTI") or "ARTI",
        metadata={"description": "ScienceON default target"},
    )
    scienceon_default_row_count: int = field(
        default=_int_env("SCIENCEON_DEFAULT_ROW_COUNT", 20),
        metadata={"description": "ScienceON default row count", "range": [1, 100]},
    )
    fulltext_target_count: int = field(
        default=_int_env("FULLTEXT_TARGET_COUNT", 30),
        metadata={"description": "Max papers after fulltext filter", "range": [15, 60]},
    )
    bm25_top_k: int = field(
        default=_int_env("BM25_TOP_K", 50),
        metadata={"description": "BM25 1st stage filter count", "range": [10, 100]},
    )
    reranker_top_k: int = field(
        default=_int_env("RERANKER_TOP_K", 15),
        metadata={"description": "LLM Reranker 2nd stage selection count", "range": [5, 50]},
    )
    rerank_models: str = field(
        default=_str_env("RERANK_MODELS", "auto") or "auto",
        metadata={"description": "Model tier: 'light' (MiniLM, CPU최적), 'full' (SPECTER2+BGE), 'auto' (GPU→full, CPU→light)"},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
        defaults = cls()

        return cls(
            tavily_max_results=int(
                configurable.get("tavily_max_results", defaults.tavily_max_results)
            ),
            arxiv_max_docs=int(
                configurable.get("arxiv_max_docs", defaults.arxiv_max_docs)
            ),
            scienceon_client_id=configurable.get(
                "scienceon_client_id", defaults.scienceon_client_id
            ),
            scienceon_mac_address=configurable.get(
                "scienceon_mac_address", defaults.scienceon_mac_address
            ),
            scienceon_key=configurable.get(
                "scienceon_key", defaults.scienceon_key
            ),
            scienceon_default_target=str(
                configurable.get(
                    "scienceon_default_target", defaults.scienceon_default_target
                )
            ),
            scienceon_default_row_count=int(
                configurable.get(
                    "scienceon_default_row_count", defaults.scienceon_default_row_count
                )
            ),
            fulltext_target_count=int(
                configurable.get("fulltext_target_count", defaults.fulltext_target_count)
            ),
            bm25_top_k=int(
                configurable.get("bm25_top_k", defaults.bm25_top_k)
            ),
            reranker_top_k=int(
                configurable.get("reranker_top_k", defaults.reranker_top_k)
            ),
            rerank_models=str(
                configurable.get("rerank_models", defaults.rerank_models)
            ),
        )
