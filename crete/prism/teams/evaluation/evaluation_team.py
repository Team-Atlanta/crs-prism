from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from crete.atoms.detection import Detection
from crete.agent.context import AgentContext
from crete.prism.states.common_state import PatchStatus
from crete.prism.states.evaluation_team_state import (
    EvaluationTeamState,
)
from crete.prism.states.inter_team_state import (
    InterTeamState,
    TeamStatus,
)
from crete.prism.teams.base_team import BaseTeam
from crete.prism.teams.evaluation.evaluation_reporter import (
    EvaluationReporter,
)
from crete.prism.teams.evaluation.evaluator import Evaluator


class EvaluationTeam(BaseTeam):
    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(llm)
        self.evaluator = Evaluator(llm)
        self.evaluation_reporter = EvaluationReporter(llm)

        self.graph_builder = StateGraph(EvaluationTeamState)
        self.compile()

    def set_context_and_detection(
        self, context: AgentContext, detection: Detection
    ) -> None:
        self.evaluator.set_context_and_detection(context, detection)

    def __call__(self, state: InterTeamState) -> dict[str, Any]:
        if state.team_status != TeamStatus.EVALUATE:
            raise ValueError(f"Invalid team status: {state.team_status}.")

        evaluation_team_state = self.compiled_graph.invoke(
            EvaluationTeamState.from_common_state(state)
        )
        return {
            "patch_status": evaluation_team_state["patch_status"],
            "evaluation_report": evaluation_team_state["evaluation_report"],
            "issue": evaluation_team_state["issue"],
        }

    def compile(self) -> None:
        self.graph_builder.add_node("evaluator", self.evaluator)  # type: ignore
        self.graph_builder.add_node("evaluation_reporter", self.evaluation_reporter)  # type: ignore

        self.graph_builder.add_edge(START, "evaluator")
        self.graph_builder.add_conditional_edges("evaluator", self.route)
        self.graph_builder.add_edge("evaluation_reporter", END)
        self._compiled_graph = self.graph_builder.compile()  # type: ignore

    def route(
        self, state: EvaluationTeamState
    ) -> Literal["evaluation_reporter", "__end__"]:
        if state.patch_status == PatchStatus.SOUND:
            return "__end__"
        if state.patch_status == PatchStatus.UNKNOWN:
            return "__end__"
        return "evaluation_reporter"

    def set_llm(self, llm: BaseChatModel) -> None:
        self.evaluator.llm = llm
        self.evaluation_reporter.llm = llm
