import re

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from crete.state.patch_state import (
    CodeSnippet,
)
from crete.prism.states.common_state import (
    CommonState,
)

CELL_CODE_RANGE_REGEX = re.compile(r"(?:<code_range>)([\s\S]*?)(?:<\/code_range>)")
CELL_ANALYSIS_REGEX = re.compile(r"(?:<analysis>)([\s\S]*?)(?:<\/analysis>)")


class AnalysisCell(BaseModel):
    code_snippets: list[CodeSnippet] = []
    analysis: str = ""

    @classmethod
    def from_str(cls, cell_content: str) -> "AnalysisCell | None":
        line_ranges = CELL_CODE_RANGE_REGEX.findall(cell_content)
        if len(line_ranges) == 0:
            return None

        code_snippets: list[CodeSnippet] = []
        for line_range in line_ranges:
            if not isinstance(line_range, str):
                continue
            try:
                file_path, line_range = line_range.split(":")
            except ValueError:
                continue
            try:
                line_start_str, line_end_str = line_range.split("-")
                line_start = int(line_start_str)
                line_end = int(line_end_str)
            except ValueError:
                continue
            if line_start < 1 or line_end < 1 or line_start > line_end:
                continue
            code_snippets.append(
                CodeSnippet(
                    file_path=file_path, line_start=line_start, line_end=line_end
                )
            )

        analysis_matches = CELL_ANALYSIS_REGEX.findall(cell_content)
        if len(analysis_matches) == 0:
            return None
        analysis = "\n".join(
            [am.strip() for am in analysis_matches if isinstance(am, str) and am != ""]
        )
        if len(code_snippets) == 0 or analysis == "":
            return None
        return AnalysisCell(code_snippets=code_snippets, analysis=analysis)

    def to_str(self, add_analysis: bool = True, add_cell_tags: bool = False) -> str:
        code_snippets = [CodeSnippet(**cs.model_dump()) for cs in self.code_snippets]
        for cs in code_snippets:
            if len(cs.content) > 0 and cs.content[-1] == "\n":
                cs.content = cs.content[:-1]
        has_empty_code_snippets = any(cs.content == "" for cs in code_snippets)
        if has_empty_code_snippets:
            formatted_code_snippets = [
                f"<code_range>{cs.file_path}:{cs.line_start}-{cs.line_end}</code_range>"
                for cs in code_snippets
            ]
        else:
            formatted_code_snippets = [
                f"<code>\n{cs.file_path}:{cs.line_start}-{cs.line_end}\n"
                f"```\n{cs.content}\n```\n</code>"
                for cs in code_snippets
            ]
        if add_analysis:
            cell_str = f"{'\n'.join(formatted_code_snippets)}\n<analysis>\n{self.analysis}\n</analysis>"
        else:
            cell_str = "\n".join(formatted_code_snippets)
        if add_cell_tags:
            cell_str = f"<cell>\n{cell_str}\n</cell>"
        return cell_str


class AnalysisTeamState(CommonState):
    messages: list[BaseMessage] = []
    cells: list[AnalysisCell] = []
    n_fix_strategy_tries: int = 0
