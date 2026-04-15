from typing import Any, Iterator, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from crete.agent.context import AgentContext
from crete.atoms.action import Action, NoPatchAction
from crete.atoms.detection import Detection
from crete.prism.states.common_state import PatchStatus
from crete.prism.states.inter_team_state import InterTeamState, TeamStatus
from crete.prism.teams import AnalysisTeam, EvaluationTeam, PatchTeam


class PrismAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        backup_llm: BaseChatModel | None = None,
        recursion_limit: int = 256,
        max_n_evals: int = 3,
    ) -> None:
        self._llm = llm
        self._backup_llm = backup_llm
        self.recursion_limit = recursion_limit
        self.max_n_evals = max_n_evals

        self.supervisor: Supervisor | None = None
        self.analysis_team: AnalysisTeam | None = None
        self.evaluation_team: EvaluationTeam | None = None
        self.patch_team: PatchTeam | None = None

        self.graph_builder = StateGraph(InterTeamState)
        self._compiled_graph = None

        self.compile(llm=self._llm)

    def act(self, context: AgentContext, detection: Detection) -> Iterator[Action]:
        if self.evaluation_team is None:
            raise ValueError("Evaluation team not initialized.")

        self.evaluation_team.set_context_and_detection(context, detection)
        if self.supervisor is not None:
            self.supervisor.set_context(context)

        final_diff = None
        try:
            patch_state = self.compiled_graph.invoke(
                InterTeamState(repo_path=str(context["pool"].patch_directory)),
                {"recursion_limit": self.recursion_limit},
            )

            final_diff = patch_state["diff"]
        except Exception as e:  # pylint: disable=broad-except
            context["logger"].warning(
                f"Error occurred while generating patch: {e}", exc_info=True
            )
            final_diff = self._patch_with_backup_llm(context)

        if isinstance(final_diff, str) and final_diff.strip() != "":
            yield context["evaluator"].evaluate(
                context, bytes(final_diff, "utf-8"), detection
            )
        else:
            yield NoPatchAction()

    @property
    def compiled_graph(self) -> CompiledStateGraph:
        if self._compiled_graph is None:
            raise ValueError("Graph not compiled. Please call compile() first.")
        return self._compiled_graph

    def compile(self, llm: BaseChatModel) -> None:
        self.supervisor = Supervisor(max_n_evals=self.max_n_evals)
        self.analysis_team = AnalysisTeam(llm=llm)
        self.evaluation_team = EvaluationTeam(llm=llm)
        self.patch_team = PatchTeam(llm=llm)

        self.graph_builder.add_node("supervisor", self.supervisor)  # type: ignore
        self.graph_builder.add_node("analysis_team", self.analysis_team.compiled_graph)  # type: ignore
        self.graph_builder.add_node(  # type: ignore
            "evaluation_team", self.evaluation_team.compiled_graph
        )
        self.graph_builder.add_node("patch_team", self.patch_team.compiled_graph)  # type: ignore

        self.graph_builder.add_edge(START, "supervisor")
        self.graph_builder.add_edge("analysis_team", "supervisor")
        self.graph_builder.add_edge("evaluation_team", "supervisor")
        self.graph_builder.add_edge("patch_team", "supervisor")
        self.graph_builder.add_conditional_edges("supervisor", self.supervisor.route)
        self._compiled_graph = self.graph_builder.compile()  # type: ignore

    def _patch_with_backup_llm(self, context: AgentContext) -> str | None:
        final_diff = None
        if self._backup_llm is None:
            return final_diff
        if (
            self.analysis_team is None
            or self.evaluation_team is None
            or self.patch_team is None
        ):
            raise ValueError("Teams not initialized. Please call compile() first.")
        context["logger"].info("Prism patching with backup LLM...")
        try:
            self.analysis_team.set_llm(self._backup_llm)
            self.evaluation_team.set_llm(self._backup_llm)
            self.patch_team.set_llm(self._backup_llm)

            patch_state = self.compiled_graph.invoke(
                InterTeamState(repo_path=str(context["pool"].patch_directory)),
                {"recursion_limit": self.recursion_limit},
            )
            final_diff = patch_state["diff"]
        except Exception as e:  # pylint: disable=broad-except
            context["logger"].error(
                f"Error occurred while generating patch(backup): {e}",
                exc_info=True,
            )
        return final_diff


class Supervisor:
    def __init__(self, max_n_evals: int = 4) -> None:
        self.max_n_evals = max_n_evals
        self.context: AgentContext | None = None

    def set_context(self, context: AgentContext) -> None:
        self.context = context

    def __call__(self, state: InterTeamState) -> dict[str, Any]:
        next_team_status = TeamStatus.END
        if state.team_status == TeamStatus.START:
            next_team_status = TeamStatus.EVALUATE
        elif state.team_status == TeamStatus.EVALUATE:
            next_team_status = TeamStatus.ANALYZE
        elif state.team_status == TeamStatus.ANALYZE:
            next_team_status = TeamStatus.PATCH
        elif state.team_status == TeamStatus.PATCH:
            next_team_status = TeamStatus.EVALUATE

        if state.team_status == TeamStatus.EVALUATE:
            state.n_evals += 1

        if state.n_evals >= self.max_n_evals:
            next_team_status = TeamStatus.END
        elif state.patch_status in (PatchStatus.SOUND, PatchStatus.UNKNOWN):
            next_team_status = TeamStatus.END

        if self.context is not None and "logger" in self.context:
            self.context["logger"].info(
                f"Prism Supervisor Routing: {state.team_status} -> {next_team_status}\n"
                + f"(n evals: {state.n_evals}, patch status: {state.patch_status})"
            )
        return {"team_status": next_team_status, "n_evals": state.n_evals}

    def route(
        self, state: InterTeamState
    ) -> Literal["analysis_team", "evaluation_team", "patch_team", "__end__"]:
        if state.team_status == TeamStatus.EVALUATE:
            return "evaluation_team"
        elif state.team_status == TeamStatus.ANALYZE:
            return "analysis_team"
        elif state.team_status == TeamStatus.PATCH:
            return "patch_team"
        return "__end__"
