import inspect
import os
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from crete.retriever.code_retriever_subgraph import (
    CodeRetrieverSubgraph,
)
from crete.prism.states.analysis_team_state import (
    AnalysisCell,
    AnalysisTeamState,
)
from crete.prism.teams.base_agent import BaseAgent


class CodeContextProvider(BaseAgent):
    system_prompt = inspect.cleandoc(
        """\
        You are an expert code context provider tasked with building a comprehensive, self-contained analysis notebook for a software issue.
        You will be given an evaluation report describing the issue in a codebase.
        Using evaluation report as a starting point, you will explore the codebase to gather relevant code snippets and analyze them to incrementally build a notebook.
        The final notebook will contain a series of cells that provide code snippets and analyses to help strategize a patch for the issue.
        ---
        You will incrementally build the notebook by:
        1. Providing an exploration plan that describes what parts of the codebase you will explore and why.
        2. Exploring the codebase to find relevant code snippets.
        3. Generating code snippets and analyses in the form of cells.

        You will iterate on the above steps until you have built a comprehensive notebook that provides a complete analysis of the issue.
        Here are some important notes to keep in mind:
        - Explore deep into the codebase to provide a comprehensive analysis.
          - Try to find similar logics and patterns that are already implemented in the codebase.
          - Try to find relevant control and data flows that must be followed.
          - Try to find and understand codebase specific constants and patterns.
          - Try to find multiple possible root causes of the issue not just the symptoms.
            - The real root cause can be hidden from the evaluation report.
            - Utility functions or helper functions are unlikely be the root cause.
            - Follow the data flow and control flow to find multiple possible root causes.
            - Explore the codebase structure to reach entirely different parts of the codebase.
            - If the symptoms or root causes are too vague, try to explore the codebase to find more concrete symptoms or root causes.
          - External libraries, fuzzer and harness code, and test code are out of scope for patching.
            - The harness code is not the root cause and it serves as an entry point for the fuzzer to test the codebase.
            - The fuzzing can be done by some APIs which are not the root cause of the issue. It is just a way to trigger the issue in the codebase.
            - Fuzzer code usually contains names like fuzz, harness, jazzer, etc.
            - Try to find the real root cause in the codebase not just traversing the stack trace.
            - Try to analyze the core functionality of the codebase that must be retained to pass the existing tests.
        - The evaluation report combined with all the cells must be self-contained.
          - You must generate the cells with analyses to build a comprehensive notebook.
          - They must include all the information needed to understand the issue.
          - They should contain repository level knowledge that is helpful to reason about the issue.
          - Try to find and analyze all the parts mentioned in the evaluation report.
          - If a failed patch attempt is present in the evaluation report, try to investigate completely different parts of the codebase.
            - The failed patch attempt may address the real root cause of the issue.
            - Try to find more relevant code snippets that are not related to the failed patch attempt.
          - If multiple issues are raised in the evaluation report, analyze all of them to build the notebook.
          - Do not try to analyze any of these code since they cannot be changed.
          - External code may not be reached by the codebase exploration.
        ---
        Below are the detailed instructions for each step.

        ### Step 1: Exploration Plan
        - Enclose the exploration plan in <exploration_plan> and </exploration_plan> tags like this:
        <exploration_plan>
        ...
        </exploration_plan>
        - Provide a concise exploration plan that describes what parts of the codebase you will explore next and why regarding the current notebook.
        - Try to grasp a holistic view of the codebase and identify the relevant parts to explore.
        - Strategize your exploration plan to effectively gather relevant code snippets from the current notebook.
        - If no further exploration is needed, write "No further exploration needed." like this:
        <exploration_plan>
        No further exploration needed.
        </exploration_plan>

        ### Step 2: Code Exploration
        - Explore the codebase by listing files or requesting code snippets using retrieval queries.
        - List files by enclosing the directory path in <list_dir> and </list_dir> tags like <list_dir>path/to/dir</list_dir>.
          - You can provide multiple <list_dir> tags to list multiple directories at once.
          - The directory path must be relative to the root directory of the codebase.
        - Request code snippets by enclosing the retrieval queries in <retrieve> and </retrieve> tags.
          - The types of retrieval queries are as follows:
            a) Querying with <grep> and </grep> tags.
              - This retrieves exact matches of the query from the codebase. (Beware that it uses a modified regex search.)
              - This can retrieve definitions or line surroundings of the query.
              - Class name, function name, variable name, type name, or a single line regex can be used as a query.
              - Examples of valid grep queries:
                <grep>example.using.fully.qualified.ClassName</grep>
                <grep>function_name</grep>
                <grep>example.using.fully.qualified.ClassName.methodName</grep>
                <grep>variable_type_name</grep>
                <grep>one_line_code</grep>
            b) Querying with <file> and </file> tags.
              - This retrieves the file content using the requested file path or name.
              - Add a line range to retrieve specific parts of the file content.
              - Examples of valid file queries:
                <file>relative/path/to/file_name.py</file>
                <file>relative/path/to/file_name.py:20-30</file>
          - The rules when requesting retrieval queries are as follows:
            - Try to request between 3 to 5 queries.
            - Since file queries can retrieve large code snippets, prefer to use grep queries.
            - Limit the number of file queries to 1.
            - Refrain from requesting too generic queries that may result too many irrelevant code snippets.
            - Prefer to use the fully qualified class name and class method name to reduce ambiguity.
        - You can choose to explore the codebase in a top-down or bottom-up manner.
        - Top-down exploration is preferred to understand the codebase structure and dependencies.
        - Bottom-up exploration is preferred to understand the code logic and data flow.
        - If no further exploration is needed, do not list any files or request any code snippets.

        ### Step 3: Cell Generation
        - Add multiple cells to the notebook by generating cells inside the <cells> and </cells> tags like this:
        <cells>
        <cell>
        ...
        </cell>
        <cell>
        ...
        </cell>
        </cells>
        - Enclose each cell in <cell> and </cell> tags with the following contents:
          a) A set of code line ranges that are relevant to the cell enclosed in <code_range> and </code_range> tags.
            - Specify the analyzing code line ranges like <code_range>path/to/file_name.py:line_start-line_end</code_range>.
            - Specify only the line ranges without any real code snippets. (Multiple line ranges are allowed.)
            - Also specify the line ranges of relevant import statements to understand the dependencies.
            - Try to select the line ranges large enough to provide more complete context.
            - Try to keep the line ranges under 200 lines. (Do not try to analyze a large file at once.)
            - Do not estimate the line ranges. Just provide the line ranges that are already retrieved.
            - If line ranges are not retrieved yet, explore the codebase to retrieve them.
          b) An analysis of the selected code line ranges enclosed in <analysis> and </analysis> tags.
            - Analyze if the symptoms described in the evaluation report are present in the selected code line ranges.
            - Analyze if potential root cause of the symptoms are present in the selected code line ranges.
            - Analyze if constants or patterns that can be used to understand or fix the issue are present in the selected code line ranges.
            - Analyze if reference code snippets that already addressed similar issues are present in the selected code line ranges.
            - Analyze if the selected code line ranges raise any questions or concerns that are relevant to the issue.
            - Try to keep the analysis concise and to the point.
            - Only one analysis is allowed per cell. (If multiple analyses are needed, create multiple cells.)
        - If there are no relevant code snippets to analyze, generate <cells> and </cells> tags without any cells like <cells></cells>.
        - If the notebook building is completed, write "Completed building the notebook." in the <cells> and </cells> tags.
        ---
        Below is an example of the output format you can follow regarding all the steps:
        <exploration_plan>
        ...
        </exploration_plan>
        <list_dir>path/to/dir_1</list_dir>
        <list_dir>path/to/dir_2</list_dir>
        <retrieve>
        <grep>function_name_1</grep>
        <grep>function_name_2</grep>
        <grep>variable_type_name</grep>
        </retrieve>
        <cells>
        <cell>
        <code_range>path/to/file_name_1.py:1-10</code_range>
        <code_range>path/to/file_name_1.py:50-100</code_range>
        <analysis>
        ...
        </analysis>
        </cell>
        <cell>
        <code_range>path/to/file_name_2.py:25-75</code_range>
        <analysis>
        ...
        </analysis>
        </cell>
        </cells>

        Here are the important rules to follow for the output format:
        - Do NOT indent the tags itself.
        - Do NOT forget to provide the <exploration_plan> and </exploration_plan> tags at every interaction.
        - If exploration is needed, always provide the retrieval queries or list directory queries in the correct format.
        - Do NOT forget to provide the <cells> and </cells> tags at every interaction.
        - Do NOT forget to provide related import statements in the <code_range> and </code_range> tags.
        - Do NOT forget to add a set of code line ranges and an analysis in the <cell> and </cell> tags.
        """
    )
    initial_user_prompt = inspect.cleandoc(
        """\
        Here is the evaluation report:
        <evaluation_report>
        {evaluation_report}
        </evaluation_report>

        Here is the root directory structure of the codebase:
        <directory_structure>
        {directory_structure}
        </directory_structure>

        Here are the additional notes while building the notebook:
        - Try to analyze the original core logic that must be retained.
        - Try to provide all the code snippets needed to resolve the issues described in the evaluation report.
        - Always try to find the root cause of the issue in the codebase other than fuzzer or harness related codes.
        - Always try to provide the code snippets that can be referenced to retain the original core logic.
        ---
        Notebook status: Building
        Number of cells: {n_cells}
        """
    )
    initial_user_prompt_with_code = inspect.cleandoc(
        """\
        Here is the evaluation report:
        <evaluation_report>
        {evaluation_report}
        </evaluation_report>

        Here is the root directory structure of the codebase:
        <directory_structure>
        {directory_structure}
        </directory_structure>

        Here are the possibly relevant code snippets you should use to build the notebook:
        <code_snippets>
        {code_snippets}
        </code_snippets>

        Here are the additional notes while building the notebook:
        - Try to analyze the original core logic that must be retained.
        - Try to provide all the code snippets needed to resolve the issues described in the evaluation report.
        - Always try to find the root cause of the issue in the codebase other than fuzzer or harness related codes.
        - Always try to provide the code snippets that can be referenced to retain the original core logic.
        ---
        Notebook status: Building.
        Number of cells: {n_cells}
        """
    )
    user_prompt = inspect.cleandoc(
        """\
        {retrieval_prompt}
        ---
        Notebook status: Building
        Number of cells: {n_cells}
        Add additional cells to the notebook using the instructed format if needed.
        """
    )
    continue_building_prompt = inspect.cleandoc(
        """\
        You are currently not exploring the codebase and only building the notebook.
        If you completed building the notebook, write "Completed building the notebook." in the <cells> and </cells> tags.  
        ---
        Notebook status: Building
        Number of cells: {n_cells}
        """
    )
    format_error_prompt = inspect.cleandoc(
        """\
        You have not generated any exploration plan inside <exploration_plan> and </exploration_plan>
        or valid cells inside <cells> and </cells> tags or retrieval queries in the correct format.
        If notebook building is not completed, provide those tags and retrieval queries in the correct format.
        Otherwise, write "Completed building the notebook." in the <cells> and </cells> tags.  
        Please regenerate to follow the format strictly.
        ---
        Notebook status: Building
        Number of cells: {n_cells}
        """
    )
    completed_but_no_cells_prompt = inspect.cleandoc(
        """\
        You have not generated any valid cells inside <cells> and </cells> tags in the correct format, but marked the notebook building as completed.
        Please regenerate the cells to follow the format strictly.
        ---
        Notebook status: Building
        Number of cells: {n_cells}
        """
    )
    list_dir_error_prompt = "Your requested directory does not exist or is empty. Please check the path and try again."
    list_dir_query_and_result_prompt = inspect.cleandoc(
        """\
        <list_dir_result>
        <list_dir>{query_path}</list_dir>
        {result}
        </list_dir_result>
        """
    )

    cell_regex = re.compile(r"(?:<cell>)([\s\S]*?)(?:<\/cell>)")
    grep_regex = re.compile(r"(?:<grep>)([\s\S]*?)(?:<\/grep>)")
    file_regex = re.compile(r"(?:<file>)([\s\S]*?)(?:<\/file>)")
    list_dir_regex = re.compile(r"(?:<list_dir>)([\s\S]*?)(?:<\/list_dir>)")

    def __init__(
        self,
        llm: BaseChatModel,
        max_n_interactions: int = 64,
        max_n_retries: int = 3,
        max_retrievals_per_query: int = 64,
        add_line_numbers: bool = True,
    ) -> None:
        super().__init__(llm)
        self.max_n_interactions = max_n_interactions
        self.max_n_retries = max_n_retries
        self.max_retrievals_per_query = max_retrievals_per_query
        self.add_line_numbers = add_line_numbers
        self.code_retriever_subgraph = CodeRetrieverSubgraph(
            max_retrievals_per_query=max_retrievals_per_query,
            add_line_numbers=add_line_numbers,
        )

    def __call__(self, state: AnalysisTeamState) -> dict[str, Any]:
        if state.evaluation_report == "":
            raise ValueError("Evaluation report must be provided")
        if self.max_n_interactions <= 0:
            raise ValueError("Max number of interactions must be greater than 0")

        if state.relevant_code_snippets == "":
            user_prompt = self.initial_user_prompt.format(
                evaluation_report=state.evaluation_report,
                directory_structure=self._get_list_dir_result("", state.repo_path),
                n_cells=0,
            )
        else:
            user_prompt = self.initial_user_prompt_with_code.format(
                evaluation_report=state.evaluation_report,
                directory_structure=self._get_list_dir_result("", state.repo_path),
                code_snippets=state.relevant_code_snippets,
                n_cells=0,
            )
        state.messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        for _ in range(self.max_n_interactions):
            state.messages.append(self.llm.invoke(state.messages))
            last_content = self._check_content_is_str(
                state.messages[-1].content  # type: ignore
            )
            for _ in range(self.max_n_retries):
                if "<cells>" in last_content or "<exploration_plan>" in last_content:
                    break
                state.messages.append(
                    HumanMessage(
                        content=self.format_error_prompt.format(
                            n_cells=len(state.cells)
                        )
                    )
                )
                state.messages.append(self.llm.invoke(state.messages))
                last_content = self._check_content_is_str(
                    state.messages[-1].content  # type: ignore
                )
            cells_to_extend = self._cells_from_message_content(last_content)
            state.cells.extend(cells_to_extend)
            if "Completed building the notebook" in last_content:
                if len(state.cells) == 0:
                    user_prompt = self.completed_but_no_cells_prompt.format(
                        n_cells=len(state.cells)
                    )
                else:
                    break
            retrieval_prompt = ""
            if "No further exploration needed" not in last_content:
                retrieval_prompt = self._list_dir_and_retrieve_code_snippets(
                    last_content, state.repo_path
                )
            if retrieval_prompt == "":
                if len(cells_to_extend) == 0:
                    user_prompt = self.format_error_prompt.format(
                        n_cells=len(state.cells)
                    )
                else:
                    user_prompt = self.continue_building_prompt.format(
                        n_cells=len(state.cells)
                    )
            else:
                user_prompt = self.user_prompt.format(
                    retrieval_prompt=retrieval_prompt,
                    n_cells=len(state.cells),
                )
            state.messages.append(HumanMessage(content=user_prompt))
        return {"cells": state.cells}

    def _cells_from_message_content(self, content: str) -> list[AnalysisCell]:
        cells: list[AnalysisCell] = []
        cell_contents = self.cell_regex.findall(content)
        if len(cell_contents) == 0:
            return cells
        for cell_content in cell_contents:
            cell = AnalysisCell.from_str(cell_content)
            if cell is None:
                continue
            cells.append(cell)
        return cells

    def _list_dir_and_retrieve_code_snippets(self, content: str, repo_path: str) -> str:
        if "</exploration_plan>" in content:
            content = content.split("</exploration_plan>")[-1]
        if "<cells>" in content:
            content = content.split("<cells>")[0]

        list_dir_prompts: list[str] = []
        list_dir_queries = self.list_dir_regex.findall(content)
        for query_path in list_dir_queries:
            list_dir_result = self._get_list_dir_result(query_path, repo_path)
            if list_dir_result == "":
                list_dir_result = self.list_dir_error_prompt
            list_dir_prompts.append(
                self.list_dir_query_and_result_prompt.format(
                    query_path=query_path, result=list_dir_result
                )
            )
            content = content.replace(f"<list_dir>{query_path}</list_dir>", "")
        list_dir_prompts_concat = "\n".join(list_dir_prompts)

        retrieved_code_snippets = self.code_retriever_subgraph.retrieve_from_content(
            content, repo_path
        )

        retrieval_prompt = ""
        if list_dir_prompts_concat != "":
            retrieval_prompt = list_dir_prompts_concat
        if retrieved_code_snippets != "":
            retrieval_prompt += "\n" + retrieved_code_snippets
        return retrieval_prompt

    def _get_list_dir_result(self, query_path: str, repo_path: str) -> str:
        rebased_query_path = self._rebase_query_path(query_path, repo_path)
        if rebased_query_path is None:
            return ""

        full_path = os.path.join(repo_path, rebased_query_path)
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            # NOTE: This should be already handled in the _rebase_query_path
            return ""

        listed_paths = [
            os.path.join(rebased_query_path, p) for p in os.listdir(full_path)
        ]
        path_strs: list[str] = []
        for path in listed_paths:
            if os.path.isdir(os.path.join(repo_path, path)):
                path_strs.append(os.path.join(f"{path}", ""))
            else:
                path_strs.append(f"{path}")
        return "\n".join(path_strs)

    def _rebase_query_path(self, query_path: str, repo_path: str) -> str | None:
        query_paths = query_path.split(os.path.sep)
        rebased_query_path = None
        for i in range(len(query_paths)):
            curr_path = os.path.join(*query_paths[i:])
            check_path = os.path.join(repo_path, curr_path)
            if os.path.exists(check_path):
                rebased_query_path = curr_path
                break
        return rebased_query_path
