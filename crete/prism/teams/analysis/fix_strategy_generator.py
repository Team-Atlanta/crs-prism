import inspect
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from crete.retriever.file_retriever import (
    FileRetriever,
    RetrievalCategory,
    RetrievalQuery,
)
from crete.state.patch_state import (
    CodeSnippet,
)
from crete.prism.states.analysis_team_state import (
    AnalysisCell,
    AnalysisTeamState,
)
from crete.prism.teams.base_agent import BaseAgent


class FixStrategyGenerator(BaseAgent):
    system_prompt = inspect.cleandoc(
        """\
        You are a fix strategy generator.
        You will be provided with an evaluation report and a set of relevant code snippets with analysis.
        Your task is to generate the best strategy to fix the issue described in the evaluation report using the relevant code snippets.
        ---
        ## Input Format

        1. Evaluation Report
        - The evaluation report is enclosed in <evaluation_report> and </evaluation_report> tags.
        - The evaluation report contains the description of the current issue.

        2. Relevant Code Snippets
        - The relevant code snippets are enclosed in <code_snippets> and </code_snippets> tags.
        - Mutiple code snippets and analysis for them are grouped by <cell> and </cell> tags.
        - Each code snippet is enclosed in <code> and </code> tags.
        - The analysis of the code snippets in each cell is enclosed in <analysis> and </analysis> tags.
        ---
        ## Output Instructions

        Generate your fix strategy inside <fix_strategy> and </fix_strategy> tags following these steps:

        1. Analyze the code snippets in relation to the evaluation report.
        - Extract core insights from the code snippets.
        - Locate symptoms and root causes in the code.
        - Identify critical logic that must be preserved to not break existing functionality.
        - Determine if previous patches should be reapplied, since the patch must be applied to the initial codebase.
        - If previous issues persist, analyze the code snippets to find alternative approaches.
        - If additional issue is reported only on the harness code, this means that the last patch attempt did not resolve the issue correctly.

        2. Generate a comprehensive fix strategy that addresses the issues in the evaluation report. (No actual code diffs are needed.)
        - Describe steps to implement the patches from the initial codebase.
        - Never generate a fix strategy that patches external libraries or dependencies.
        - Never generate a fix strategy that modifies fuzzing logic or harness codes.
          - A fix strategy modifying the harness code is strictly forbidden. The harness code is not the root cause of the issue.
          - The harness code is not the root cause and it serves as an entry point for the fuzzer to test the codebase.
          - The fuzzing can be done by some APIs which are not the root cause of the issue. It is just a way to trigger the issue in the codebase.
          - The fix strategy must support the same fuzzing logic. (Modifying fuzzing related functions, macros, or flags are strictly forbidden.)
          - Harness code usually contains names like "fuzz", "jazz", "harness", etc. They are not part of the source code to be modified.
        - Never generate a fix strategy that patches any test codes. The patched code must not break the existing tests.
        - Avoid modifying utility functions that might affect the broader codebase.
        - Avoid removing a functionality without providing an alternative solution. (Removal only patch is only allowed for mallicious code.)
        - Avoid catching too broad exceptions or errors that might hide underlying issues.
        - For compilation errors, focus on the line numbers, variable names, and logical flow.

        3. Explain your fix strategy.
        - Justify why this is the optimal approach.
        - Explain how it preserves existing codebase logic.
        - Clarify how it addresses the issues in the evaluation report.
        - Discuss potential side effects and how to mitigate them.
        - Reference similar solutions from the codebase if applicable.

        4. Retrieve more relevant code snippets to provide more context or update the fix strategy. (Optional)
        a) You can retrieve file contents from the codebase by querying like the following:
        <file>relative/path/to/file_name_1.py:20-30</file>
        <file>relative/path/to/file_name_2.py:1-50</file>
        - Try to limit the number of retrieved lines to 50 lines per file.
        - Try to limit the number of retrieved files to 5 files.
        - You can use this to query where import statements are located if needed.
        - Do not retrieve files that are not related to the issue.
        b) Update the fix strategy with the retrieved file contents.
        - Only update the fix strategy if the retrieved file contents must be used as the patch context.
        c) After retrieving the file contents, add them to the relevant code snippets if they are used to update the fix strategy.
        - Add the code ranges that are used to update the fix strategy like:
        <code_range>path/to/file_name_1.py:1-10</code_range>
        <code_range>path/to/file_name_1.py:50-100</code_range>
        - This must be done after retrieving the file contents. Do not assume the line ranges.         

        You can iterate retrieving code snippets, updating the fix strategy, and adding relevant code snippets until you are satisfied with the fix strategy.
        If you are satisfied with the fix strategy, you can stop iterating by not retrieving files anymore.
        """
    )
    user_prompt_empty_diff = inspect.cleandoc(
        """\
        Here is the evaluation report:
        <evaluation_report>
        {evaluation_report}
        </evaluation_report>

        Here are the relevant code snippets and analysis:
        <code_snippets>
        {code_snippets}
        </code_snippets>
        """
    )
    user_prompt_with_diff = inspect.cleandoc(
        """\
        Here is the evaluation report:
        <evaluation_report>
        {evaluation_report}
        </evaluation_report>

        Here is the last patch attempt:
        <last_patch_attempt>
        {last_patch_attempt}
        </last_patch_attempt>

        Here are the relevant code snippets and analysis:
        <code_snippets>
        {code_snippets}
        </code_snippets>
        """
    )
    format_error_prompt = inspect.cleandoc(
        """\
        You have not generated any <fix_strategy> and </fix_strategy> tags or provided invalid <file> and </file> tags.
        Here are some reasons why this might happen:
        - You have not generated any fix strategy.
        - File path between <file> and </file> tags is not a full path or does not exist.
        - Line range between <file> and </file> tags is not valid or does not exist.
        Please generate the valid tags you need.
        """
    )
    code_snippet_template = inspect.cleandoc(
        """\
        <code>
        {file_path}:{line_start}-{line_end}
        ```
        {content}
        ```
        </code>
        """
    )
    analysis_report_template = inspect.cleandoc(
        """\
        # Analysis Report

        ## Relevant Code Snippets
        <code_snippets>
        {relevant_code_snippets}
        </code_snippets>

        ## Fix Strategy
        <fix_strategy>
        {fix_strategy}
        </fix_strategy>
        """
    )
    fix_strategy_regex = re.compile(r"(?:<fix_strategy>)([\s\S]*?)(?:<\/fix_strategy>)")
    file_query_regex = re.compile(r"(?:<file>)([\s\S]*?)(?:<\/file>)")
    code_range_regex = re.compile(r"(?:<code_range>)([\s\S]*?)(?:<\/code_range>)")

    def __init__(
        self, llm: BaseChatModel, max_n_cells: int = 256, max_n_interactions: int = 6
    ) -> None:
        super().__init__(llm)
        self.max_n_cells = max_n_cells
        self.max_n_interactions = max_n_interactions
        self.file_retriever = FileRetriever(
            add_line_numbers=True,
            max_n_results_per_query=8,
        )

    def __call__(self, state: AnalysisTeamState) -> dict[str, Any]:
        if state.evaluation_report == "":
            raise ValueError("Evaluation report must be provided")
        state.n_fix_strategy_tries += 1
        if len(state.cells) == 0:
            return {
                "analysis_report": "",
                "relevant_code_snippets": "",
                "n_fix_strategy_tries": state.n_fix_strategy_tries,
            }

        valid_cells = self._retrieve_cell_codes(state.cells, state.repo_path)
        formatted_cell_prompts = self._format_cell_prompts(valid_cells)
        if formatted_cell_prompts == "":
            return {
                "analysis_report": "",
                "relevant_code_snippets": "",
                "n_fix_strategy_tries": state.n_fix_strategy_tries,
            }

        if state.diff == "":
            user_prompt = self.user_prompt_empty_diff.format(
                evaluation_report=state.evaluation_report,
                code_snippets=formatted_cell_prompts,
            )
        else:
            user_prompt = self.user_prompt_with_diff.format(
                evaluation_report=state.evaluation_report,
                last_patch_attempt=state.diff,
                code_snippets=formatted_cell_prompts,
            )

        state.messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        fix_strategy = ""
        for _ in range(self.max_n_interactions):
            state.messages.append(self.llm.invoke(state.messages))
            last_message_content = self._check_content_is_str(
                state.messages[-1].content  # type: ignore
            )
            fix_strategy_matches = self.fix_strategy_regex.findall(last_message_content)
            if len(fix_strategy_matches) > 0:
                fix_strategy = fix_strategy_matches[0]

            code_range_matches = self.code_range_regex.findall(last_message_content)
            if len(code_range_matches) > 0:
                self._add_code_ranges_to_cells(
                    valid_cells,
                    code_range_matches,
                    state.repo_path,
                )

            file_query_matches = self.file_query_regex.findall(last_message_content)
            if len(file_query_matches) == 0:
                break

            retrieved_code_snippets: list[CodeSnippet] = []
            for file_query_match in file_query_matches:
                retrieved_code_snippet = self._retieve_file_content(
                    file_query_match, state.repo_path
                )
                if retrieved_code_snippet is None:
                    continue
                retrieved_code_snippets.append(retrieved_code_snippet)
            if len(retrieved_code_snippets) == 0:
                state.messages.append(
                    HumanMessage(
                        content=self.format_error_prompt,
                    )
                )
            else:
                state.messages.append(
                    HumanMessage(
                        content="\n".join(
                            [
                                self.code_snippet_template.format(
                                    file_path=cs.file_path,
                                    line_start=cs.line_start,
                                    line_end=cs.line_end,
                                    content=cs.content,
                                )
                                for cs in retrieved_code_snippets
                            ]
                        ),
                    )
                )

        if fix_strategy == "":
            # NOTE: Last attempt to get the fix strategy
            state.messages = [
                SystemMessage(
                    content=self.system_prompt.split(
                        "After generating the fix strategy", 1
                    )[0]
                ),
                HumanMessage(content=user_prompt),
            ]

        relevant_code_snippets = self._format_code_snippets(valid_cells)
        analysis_report = self.analysis_report_template.format(
            relevant_code_snippets=relevant_code_snippets,
            fix_strategy=fix_strategy,
        )
        return {
            "analysis_report": analysis_report,
            "relevant_code_snippets": relevant_code_snippets,
            "n_fix_strategy_tries": state.n_fix_strategy_tries,
        }

    def _retieve_file_content(
        self, file_content_query: str, repo_path: str
    ) -> CodeSnippet | None:
        file_content_query = file_content_query.strip()
        if len(file_content_query) == 0:
            return None
        try:
            file_path, line_range = file_content_query.split(":")
            line_start, line_end = line_range.split("-")
            line_start = int(line_start)
            line_end = int(line_end)
        except ValueError:
            return None
        ret_query = RetrievalQuery(
            query=f"{file_path}:{line_start}-{line_end}",
            repo_path=repo_path,
            category=RetrievalCategory.FILE,
        )
        ret_results = self.file_retriever._retrieve(ret_query)  # type: ignore
        if len(ret_results) == 0:
            return None
        # NOTE: Only one code snippet is expected
        ret_result = ret_results[0]
        if ret_result.content is None or ret_result.content == "":
            return None
        if ret_result.file_path is None or ret_result.file_path == "":
            return None
        if ret_result.line_start is None or ret_result.line_start == 0:
            return None
        if ret_result.line_end is None or ret_result.line_end == 0:
            return None
        return CodeSnippet(
            repo_path=repo_path,
            file_path=ret_result.file_path,
            line_start=ret_result.line_start,
            line_end=ret_result.line_end,
            content=ret_result.content,
        )

    def _retrieve_cell_codes(
        self, cells: list[AnalysisCell], repo_path: str
    ) -> list[AnalysisCell]:
        valid_cells: list[AnalysisCell] = []
        for cell in cells:
            if cell.analysis == "":
                continue
            valid_code_snippets: list[CodeSnippet] = []
            for cs in cell.code_snippets:
                retrieved_code_snippet = self._retieve_file_content(
                    f"{cs.file_path}:{cs.line_start}-{cs.line_end}", repo_path
                )
                if retrieved_code_snippet is None:
                    continue
                valid_code_snippets.append(retrieved_code_snippet)
            if len(valid_code_snippets) == len(cell.code_snippets):
                cell.code_snippets = valid_code_snippets
                valid_cells.append(cell)
        return valid_cells

    def _format_cell_prompts(self, valid_cells: list[AnalysisCell]) -> str:
        return "\n".join(
            [cell.to_str(add_analysis=True, add_cell_tags=True) for cell in valid_cells]
        )

    def _format_code_snippets(self, valid_cells: list[AnalysisCell]) -> str:
        return "\n".join(
            [
                cell.to_str(add_analysis=False, add_cell_tags=False)
                for cell in valid_cells
            ]
        )

    def _add_code_ranges_to_cells(
        self, cells: list[AnalysisCell], file_query_matches: list[str], repo_path: str
    ) -> None:
        code_snippets: list[CodeSnippet] = []
        for file_content_query in file_query_matches:
            retrieved_code_snippet = self._retieve_file_content(
                file_content_query, repo_path
            )
            if retrieved_code_snippet is None:
                continue
            code_snippets.append(retrieved_code_snippet)
        if len(code_snippets) == 0:
            return
        cells.append(AnalysisCell(code_snippets=code_snippets))
