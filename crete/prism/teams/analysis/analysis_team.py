from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import (
    START,
    StateGraph,
)

from crete.prism.states.analysis_team_state import (
    AnalysisTeamState,
)
from crete.prism.states.inter_team_state import (
    InterTeamState,
    TeamStatus,
)
from crete.prism.teams.analysis.code_context_provider import (
    CodeContextProvider,
)
from crete.prism.teams.analysis.fix_strategy_generator import (
    FixStrategyGenerator,
)
from crete.prism.teams.base_team import BaseTeam


class AnalysisTeam(BaseTeam):
    def __init__(self, llm: BaseChatModel, max_n_fix_strategy_tries: int = 3) -> None:
        super().__init__(llm)
        self.max_n_fix_strategy_tries = max_n_fix_strategy_tries
        self.code_context_provider = CodeContextProvider(llm)
        self.fix_strategy_generator = FixStrategyGenerator(llm)

        self.graph_builder = StateGraph(AnalysisTeamState)
        self.compile()

    def __call__(self, state: InterTeamState) -> dict[str, Any]:
        if state.team_status != TeamStatus.ANALYZE:
            raise ValueError(f"Invalid team status: {state.team_status}.")
        if state.evaluation_report == "":
            raise ValueError("Evaluation report is empty.")

        analysis_team_state = self.compiled_graph.invoke(
            AnalysisTeamState.from_common_state(state)
        )
        return {
            "analysis_report": analysis_team_state["analysis_report"],
            "relevant_code_snippets": analysis_team_state["relevant_code_snippets"],
        }

    def compile(self) -> None:
        # TODO: Add multi turn analysis with routing
        self.graph_builder.add_node(  # type: ignore
            "fix_strategy_generator", self.fix_strategy_generator
        )
        self.graph_builder.add_node("code_context_provider", self.code_context_provider)  # type: ignore

        self.graph_builder.add_edge(START, "code_context_provider")
        self.graph_builder.add_edge("code_context_provider", "fix_strategy_generator")
        self.graph_builder.add_conditional_edges(
            "fix_strategy_generator", self.route_fix_strategy_generator
        )
        self._compiled_graph = self.graph_builder.compile()  # type: ignore

    def route_fix_strategy_generator(
        self, state: AnalysisTeamState
    ) -> Literal["code_context_provider", "__end__"]:
        if state.analysis_report != "" and state.relevant_code_snippets != "":
            return "__end__"
        if state.n_fix_strategy_tries >= self.max_n_fix_strategy_tries:
            raise ValueError(
                f"Max number of fix strategy tries exceeded: {self.max_n_fix_strategy_tries}"
            )
        return "code_context_provider"

    def set_llm(self, llm: BaseChatModel) -> None:
        self.code_context_provider.llm = llm
        self.fix_strategy_generator.llm = llm
