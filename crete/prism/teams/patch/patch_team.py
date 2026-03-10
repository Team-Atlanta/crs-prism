from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import (
    START,
    StateGraph,
)

from crete.prism.states.inter_team_state import (
    InterTeamState,
    TeamStatus,
)
from crete.prism.states.patch_team_state import PatchTeamState
from crete.prism.teams.base_team import BaseTeam
from crete.prism.teams.patch.patch_generator import (
    PatchGenerator,
)
from crete.prism.teams.patch.patch_reviewer import (
    PatchReviewer,
)


class PatchTeam(BaseTeam):
    def __init__(self, llm: BaseChatModel, max_n_reviews: int = 3) -> None:
        super().__init__(llm)
        self.max_n_reviews = max_n_reviews
        self.patch_generator = PatchGenerator(llm)
        self.patch_reviewer = PatchReviewer(llm)

        self.graph_builder = StateGraph(PatchTeamState)
        self.compile()

    def __call__(self, state: InterTeamState) -> dict[str, Any]:
        if state.team_status != TeamStatus.PATCH:
            raise ValueError(f"Invalid team status: {state.team_status}.")
        if state.evaluation_report == "" or state.analysis_report == "":
            raise ValueError("Evaluation report or analysis report is empty.")

        patch_team_state = PatchTeamState.from_common_state(state)
        patch_team_state = self.compiled_graph.invoke(patch_team_state)
        return {
            "diff": patch_team_state["diff"],
            "applied_patches": patch_team_state["applied_patches"],
        }

    def compile(self) -> None:
        self.graph_builder.add_node("patch_generator", self.patch_generator)  # type: ignore
        self.graph_builder.add_node("patch_reviewer", self.patch_reviewer)  # type: ignore

        self.graph_builder.add_edge(START, "patch_generator")
        self.graph_builder.add_conditional_edges(
            "patch_generator", self.route_generator
        )
        self.graph_builder.add_conditional_edges("patch_reviewer", self.route_reviewer)
        self._compiled_graph = self.graph_builder.compile()  # type: ignore

    def route_generator(
        self, state: PatchTeamState
    ) -> Literal["patch_reviewer", "__end__"]:
        if state.n_reviews >= self.max_n_reviews:
            return "__end__"
        return "patch_reviewer"

    def route_reviewer(
        self, state: PatchTeamState
    ) -> Literal["patch_generator", "__end__"]:
        if state.passed_checks:
            return "__end__"
        return "patch_generator"

    def set_llm(self, llm: BaseChatModel) -> None:
        self.patch_generator.llm = llm
        self.patch_reviewer.llm = llm
