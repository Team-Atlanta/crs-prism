import inspect
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from crete.state.patch_state import (
    format_patches_to_str,
)
from crete.prism.states.patch_team_state import PatchTeamState
from crete.prism.teams.base_agent import BaseAgent


class PatchReviewer(BaseAgent):
    # TODO: The review does not change the patch behavior. Update the prompt to fix this.
    system_prompt = inspect.cleandoc(
        """\
        You are a patch reviewer agent.
        You will be provided an evaluation report, an analysis report and the applied patches.
        Your task is to write a review of the applied patches based on the evaluation report and the analysis report.
        ---
        Here are the details of the evaluation report:
        - The evaluation report is enclosed in <evaluation_report> and </evaluation_report> tags.
        - The evaluation report contains the description of the current issue.

        Here are the details of the analysis report:
        - The analysis report is enclosed in <analysis_report> and </analysis_report> tags.
        - The analysis report contains the description of the analysis of the issue.
          - This includes code snippets and explanations of the code snippets. 

        Here are the details of the applied patches:
        - Each patch is enclosed in <patch> and </patch> tags.
        - The original code snippet is enclosed in <original_code> and </original_code> tags.
          - This is the original code snippet that is replaced by the patched code.
        - The code lines to replace are enclosed in <code_lines_to_replace> and </code_lines_to_replace> tags.
          - This is the line range of the original code snippet that is replaced by the patched code snippet.
        - The patched code that replaces the original code is enclosed in <patched_code> and </patched_code> tags.
          - This is the patched code that replaces the original code snippet.

        Here are the instructions for the patch review:
        1. Perform checks with detailed explanations enclosed in <check> and </check> tags:
        - Check if the patched code snippets can correctly replace the original code snippets.
          - Only check the difference between the original code snippets and the patched code snippets not the patched code snippets themselves.
          - When the patched code has additional bracket or removed a bracket compared to the original code, it cannot replace the original code.
          - The replacement must not potentially introduce any compilation errors.
          - The patched code must replace the logic of the original code while providing the solution to the issue.
        - Check if the patched code snippets addresses the issue described in the evaluation report.
          - There can be additional patches needed to address the issue that can be added from the analysis report.
        - Check if the patched code snippets follows the strategy described in the analysis report.
        2. Write a list of possible improvements enclosed in <improvements> and </improvements> tags:
        - If any of the checks fail, write a list of possible improvements for failing checks.
        - If there are redundant code or syntax errors, write how to change the applied patches.
          - There is a high change that this is due to the incorrect line range in the "code_lines_to_replace" in the applied patches.
          - If incorrect line range is detected, write the correct line range by comparing the line numbers in the original code snippets and the "code_lines_to_replace".
        3. Write a final verdict on the patched code snippets enclosed in <verdict> and </verdict> tags:
        - If all checks pass, write "The patches are valid."
        - If any of the checks fails, write "The patches are invalid."
        ---
        Here is an example format of a valid patch review:
        <check>
        ...
        </check>
        <improvements>
        ...
        </improvements>
        <verdict>
        The patches are valid.
        </verdict>

        Here is an example format of an invalid patch review:
        <check>
        ...
        </check>
        <improvements>
        ...
        </improvements>
        <verdict>
        The patches are invalid.
        </verdict>
        """
    )
    user_prompt = inspect.cleandoc(
        """\
        Here is the evaluation report:
        <evaluation_report>
        {evaluation_report}
        </evaluation_report>

        Here is the analysis report:
        <analysis_report>
        {analysis_report}
        </analysis_report>

        Here are the applied patches:
        {applied_patches}

        Here is the final diff from the applied patches:
        <diff>
        {diff}
        </diff>

        Please write a review of the applied patches based on the evaluation report and the analysis report.
        """
    )
    empty_patch_message = inspect.cleandoc(
        """\
        Generated patches are not valid. Here are some possible reasons why:
        - Incorrect patch is generated not adhering to the patch format.
        - Trying to patch a file that does not exist in the repository. (Fuzzer and harness codes or external libraries)
        - Trying to patch files that cannot be altered.

        Please regenerate the patch focusing on the patch format and the content of the patch.
        """
    )

    verdict_regex = re.compile(r"(?:<verdict>)([\s\S]*?)(?:<\/verdict>)")
    patch_regex = re.compile(r"(?:<patch>)([\s\S]*?)(?:<\/patch>)")
    retrieved_code_regex = re.compile(r"(?:<code>)([\s\S]*?)(?:<\/code>)")
    hunk_header_regex = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    def __call__(self, state: PatchTeamState) -> dict[str, Any]:
        state.n_reviews += 1
        if state.diff == "" or len(state.applied_patches) == 0:
            return {
                "patch_review": self.empty_patch_message,
                "passed_checks": False,
                "n_reviews": state.n_reviews,
            }

        state.messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=self.user_prompt.format(
                    evaluation_report=state.evaluation_report,
                    analysis_report=state.analysis_report,
                    applied_patches=format_patches_to_str(state.applied_patches),
                    diff=state.diff,
                )
            ),
        ]
        state.messages.append(self.llm.invoke(state.messages))
        patch_review = self._check_content_is_str(state.messages[-1].content)  # type: ignore
        verdict = self.verdict_regex.findall(patch_review)
        # TODO: Test which default value is better for empty verdict handling
        passed_checks = True
        if len(verdict) > 0 and "invalid" in verdict[0].lower():
            passed_checks = False
        return {
            "patch_review": patch_review,
            "passed_checks": passed_checks,
            "n_reviews": state.n_reviews,
        }
