import inspect
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from crete.patcher.line_range_patch_extractor import (
    LineRangePatchExtractor,
)
from crete.state.patch_state import (
    CodeDiff,
    CodeSnippet,
    format_patches_to_str,
)
from crete.prism.states.patch_team_state import PatchTeamState
from crete.prism.teams.base_agent import BaseAgent


class PatchGenerator(BaseAgent):
    system_prompt = inspect.cleandoc(
        """\
        You are a patch generator agent.
        You will be provided an evaluation report, an analysis report and the review of the previous patch if any.
        Your task is to generate patches that resolves the issue described in the evaluation report.
        ---
        Here are the details of the evaluation report:
        - The evaluation report is enclosed in <evaluation_report> and </evaluation_report> tags.
        - The evaluation report contains the description of the current issue.

        Here are the details of the analysis report:
        - The analysis report is enclosed in <analysis_report> and </analysis_report> tags.
        - The analysis report contains the description of the analysis of the issue.
          - This includes code snippets and explanations of the code snippets. 

        Here are the details of the review of the previous patch:
        - The review of the previous patch is enclosed in <patch_review> and </patch_review> tags.

        Here are the instructions and rules to follow while generating the patches:
        - Generate all the patches needed inside the <patches> and </patches> tags.
        - Enclose each patch inside <patch> and </patch> tags.
        - For each patch, follow the steps below:
          a) Copy the original code to replace from the retrieved code between <original_code> and </original_code> tags.
            - The original code must be a unique code that can be replaced by the generated patch.
            - Copy the exact line numbers, indentations, white spaces, and newlines of the original code to successfully replace it.
            - Try to copy multiple lines of code to replace with context.
            - Do NOT abbreviate the original code.
            - No overlapping original code lines are allowed when providing multiple patches.
          b) Copy the file path and code line range to replace between <code_lines_to_replace> and </code_lines_to_replace> tags.
            - The line range should be in the format of "path/to/file.py:line_start-line_end".
            - Copy the first line number of the original code to replace as line_start.
            - Copy the last line number of the original code to replace as line_end.
            - Do NOT estimate the line numbers.
            - Do NOT copy only the new line character as the start or end line itself. (White spaces with code are allowed.)
            - No overlapping line range are allowed when providing multiple patches. (e.g. 1-5, 6-10 is allowed, but 1-5, 5-10 is not allowed)
          c) Generate a patched code that can replace the copied original code inside <patched_code> and </patched_code> tags.
            - The patched code must successfully replace the original code without compilation errors.
            - Follow the same indentation level of the original code exactly in the patched code.
            - Do NOT try to replace a omitted or abbreviated code.
            - Do NOT try to abbreviate in the patched code.
            - Do NOT write line numbers in the patched code.
            - Do NOT generate an empty patched code. (In case of only removing the code, try to replace it with a comment.)
        - Repeat the process for each patch to provide multiple patches for multiple line ranges.
          - You must provide all the patches needed to address the issue. (The patch is always applied to the initial codebase.)
          - If multiple functions or classes are needed to be patched, try to split them into separate non-overlapping patches.
          - Since previous patches are not applied, provide all the patches needed to address the issue or multiple issues.
          - If the previous patches must be applied before the current patch, provide those again.
          - If incrementailly solving multiple issues, do not forget to provide all the patches including the previous ones.
          - Do not forget to patch the import statements when additional imports are needed.
        - Beware to not indent any of the tags itself.
        - Example of a single patch generation exactly replacing the code:
        <code>
        path/to/original_file.c:10-18
        ```c
        10:
        11:int add(int a, int b) {
        12:    return a - b;
        13:}
        14:
        15:int sub(int a, int b) {
        16:    return a - b;
        17:}
        18:
        ```
        </code>
        For the above example, a valid patch with exact line ranges is as follows:
        <patches>
        <patch>
        <original_code>
        ```c
        11:int add(int a, int b) {
        12:    return a - b;
        13:}
        ```
        </original_code>
        <code_lines_to_replace>
        path/to/original_file.c:11-13
        </code_lines_to_replace>
        <patched_code>
        ```c
        int add(int a, int b) {
            return a + b;
        }
        ```
        </patched_code>
        </patch>
        </patches>

        - Example of a multiple patch generation:
        <patches>
        <patch>
        <original_code>
        ```python
        111:    def class_method(self):
        112:        # multiple lines of
        113:        # original code for the issue
        114:        return
        ```
        </original_code>
        <code_lines_to_replace>
        path/to/original_file_1.py:111-114
        </code_lines_to_replace>
        <patched_code>
        ```python
            def class_method(self):
                # multiple lines of
                # patched code for the issue
                return
        ```
        </patched_code>
        </patch>
        <patch>
        <original_code>
        ```python
        333:def another_method(self):
        334:    # multiple lines of
        335:    # original code for the issue
        336:    return
        ```
        </original_code>
        <code_lines_to_replace>
        path/to/original_file_2.py:333-336
        </code_lines_to_replace>
        <patched_code>
        ```python
        def another_method():
            # patched code for
            # previous issue or
            # another location
            return
        ```
        </patched_code>
        </patch>
        </patches>
        ---
        Here are the important rules to follow while generating the patches:
        - Do not indent any of the tags itself, but match the indent exactly for the patched code to replace the original code lines.
        - Do not forget to patch the imports when additional imports are needed.
        - Do not generate a new file. Only provide patches that can be applied to the original codebase.
        - Do not generate comments in the patched code unless it only patches to remove the original code.
        - Do not modify any fuzzing or harness related codes. They are out of scope. (The file usually contains names like fuzz, harnesses, test, etc.)
        - Do not fix fuzzer or harness related files, functions or definitions to bypass the issue and not patching the root cause.
        - Do not modify or fix any test codes. The patched code must not break the existing tests and functionalities.
        - Prefer to update existing functions or classes instead of creating new ones.
        - Prefer to patch easily replaceable line ranges that does not end abruptly. (Matching brackets, etc.)
        - Always add line numbers to the original code to replace.
        - Always provide all the patches to patch from the initial codebase.
        - If multiple issues must be solved sequentially, provide all the patches to address the issues.
        """
    )
    user_prompt_empty_last_patch = inspect.cleandoc(
        """\
        Here is the evaluation report:
        <evaluation_report>
        {evaluation_report}
        </evaluation_report>

        Here is the analysis report:
        <analysis_report>
        {analysis_report}
        </analysis_report>
        """
    )
    user_prompt_with_last_patch = inspect.cleandoc(
        """\
        Here is the evaluation report:
        <evaluation_report>
        {evaluation_report}
        </evaluation_report>

        Here is the analysis report:
        <analysis_report>
        {analysis_report}
        </analysis_report>

        Here is the last patch attempt:
        {applied_patches}

        If the last patch attempt needs to be applied again according to the reports, generate the patches again.
        """
    )
    user_prompt_review = inspect.cleandoc(
        """\
        Here are the patches that can be applied to the codebase from the generated patches:
        <applied_patches>
        {applied_patches}
        </applied_patches>

        Here is the review of the generated patches:
        <patch_review>
        {patch_review}
        </patch_review>

        You must provide all the patches needed to address the issue while considering the review of the previous patch.
        Now start generating the patches enclosed in <patches> and </patches> tags.
        """
    )
    single_patch_template = inspect.cleandoc(
        """\
        <patch>
        <original_code>
        ```
        {original_code}
        ```
        </original_code>
        <code_lines_to_replace>
        {orig_file_path}:{orig_line_start}-{orig_line_end}
        </code_lines_to_replace>
        <patched_code>
        ```
        {patched_code}
        ```
        </patched_code>
        </patch>
        """
    )
    patch_regex = re.compile(r"(?:<patch>)([\s\S]*?)(?:<\/patch>)")
    original_code_regex = re.compile(
        r"(?:<original_code>)([\s\S]*?)(?:<\/original_code>)"
    )

    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(llm)
        self.patch_extractor = LineRangePatchExtractor(n_check_lines=5)

    def __call__(self, state: PatchTeamState) -> dict[str, Any]:
        if len(state.analysis_report) == 0 or len(state.evaluation_report) == 0:
            raise ValueError("Evaluation report and analysis report must be provided")

        if len(state.messages) == 0:
            if len(state.applied_patches) == 0:
                user_prompt = self.user_prompt_empty_last_patch.format(
                    evaluation_report=state.evaluation_report,
                    analysis_report=state.analysis_report,
                )
            else:
                user_prompt = self.user_prompt_with_last_patch.format(
                    evaluation_report=state.evaluation_report,
                    analysis_report=state.analysis_report,
                    applied_patches=format_patches_to_str(state.applied_patches),
                )
            state.messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt),
            ]
        else:
            if state.patch_review == "":
                raise ValueError("Patch review must be provided")
            state.messages.append(
                HumanMessage(
                    content=self.user_prompt_review.format(
                        applied_patches=format_patches_to_str(state.applied_patches),
                        patch_review=state.patch_review,
                    )
                )
            )

        state.messages.append(self.llm.invoke(state.messages))
        last_patch_attempt = self._check_content_is_str(state.messages[-1].content)  # type: ignore
        code_diff = CodeDiff()
        self._extract_and_add_patches(last_patch_attempt, code_diff, state.repo_path)
        diff = code_diff.concatenated_diff
        if diff == "" and state.diff != "":
            # NOTE: Revert back to last diff if possible
            diff = state.diff
            applied_patches = state.applied_patches
        else:
            applied_patches = code_diff.applied_patches
        return {
            "diff": diff,
            "applied_patches": applied_patches,
            "messages": state.messages,
        }

    def _extract_and_add_patches(
        self, last_message_content: str, code_diff: CodeDiff, repo_path: str
    ) -> None:
        patch_regex_matches = self.patch_regex.findall(last_message_content)
        if len(patch_regex_matches) == 0:
            return

        extracted_patches: list[CodeSnippet] = []
        for match_content in patch_regex_matches:
            patch = self.patch_extractor.extract_patch_from_content(
                repo_path, match_content
            )
            extracted_patches.append(patch)
        code_diff.add_patches(extracted_patches)
