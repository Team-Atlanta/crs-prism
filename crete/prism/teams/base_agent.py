from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


class BaseAgent(ABC):
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    @abstractmethod
    def __call__(self, state: Any) -> dict[str, Any]:
        pass  # pragma: no cover

    def _check_content_is_str(self, content: Any) -> str:
        if not isinstance(content, str):
            raise ValueError(f"Unexpected message content type: {type(content)}")
        return content
