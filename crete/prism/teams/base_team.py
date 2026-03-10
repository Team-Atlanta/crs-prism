from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import (
    CompiledStateGraph,
)


class BaseTeam(ABC):
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self._compiled_graph: CompiledStateGraph | None = None

    @property
    def compiled_graph(self) -> CompiledStateGraph:
        if self._compiled_graph is None:
            raise ValueError("Graph not compiled. Please call compile() first.")
        return self._compiled_graph

    @abstractmethod
    def __call__(self, state: Any) -> dict[str, Any]:
        pass  # pragma: no cover

    @abstractmethod
    def compile(self, *args, **kwargs) -> None:  # type: ignore
        pass  # pragma: no cover
