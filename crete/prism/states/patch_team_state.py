from langchain_core.messages import BaseMessage

from crete.prism.states.common_state import (
    CommonState,
)


class PatchTeamState(CommonState):
    messages: list[BaseMessage] = []
    patch_review: str = ""
    passed_checks: bool = False
    n_reviews: int = 0
