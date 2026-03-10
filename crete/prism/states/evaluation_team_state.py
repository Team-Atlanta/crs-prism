from langchain_core.messages import BaseMessage

from crete.prism.states.common_state import (
    CommonState,
)


class EvaluationTeamState(CommonState):
    messages: list[BaseMessage] = []
    repo_lang: str = ""
    patch_result: str = ""
    tests_log: str = ""
