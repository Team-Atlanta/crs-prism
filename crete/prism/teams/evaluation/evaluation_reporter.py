import inspect
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from crete.prism.states.evaluation_team_state import (
    EvaluationTeamState,
)
from crete.prism.teams.base_agent import BaseAgent


class EvaluationReporter(BaseAgent):
    system_prompt = inspect.cleandoc(
        """\
        You are an evaluation reporter agent.
        Your task is to generate an evaluation report based on the user provided issue of a codebase.
        """
    )
    user_prompt_issue_only = inspect.cleandoc(
        """\
        Here is the codebase issue:
        <issue>
        {issue}
        </issue>

        Based on this issue, generate an evaluation report with the following sections:
        ## Summary
        Provide a summary of the issue.

        ## Description
        Deliver a detailed description that:
        - Describes what the issue is without speculating on causes.
        - Describes the vulnerability type if mentioned in the issue.
        - Highlights important observations and findings in the issue.
        - Describes which parts of the codebase should be investigated to get a holistic view of the issue.
        ---
        Format the report as follows:
        # Evaluation Report

        ## Summary
        
        ...

        ## Description

        ...
        ---
        While generating the report, consider the following:
        - The harness code is not the root cause and it serves as an entry point for the fuzzer to test the codebase.
        - Make sure to cover all important aspects of the issue.
        - Do not speculate on the causes of the issue or provide any solutions.
        """
    )
    user_prompt_with_previous_report = inspect.cleandoc(
        """\
        Here is the initial issue of the codebase:
        <issue>
        {issue}
        </issue>

        Here is the patch attempt:
        <patch_attempt>
        {patch_attempt}
        </patch_attempt>

        Here is the result after applying the patch:
        <patch_result>
        {patch_result}
        </patch_result>

        Based on this information, generate an evaluation report with the following sections:
        ## Summary
        Provide a summary of the initial issue, the patch attempt, and the result of the patch.

        ## Description
        Deliver a detailed description that:
        - Describes what the initial issue is without speculating on causes.
        - Highlights important observations and findings in the initial issue.
        - Describes the patch attempt and the log after applying the patch detailed enough to reapply the patch.
        - Describes which parts of the codebase should be investigated further to get a holistic view of the issues.
        ---
        Format the report as follows:
        # Evaluation Report

        ## Summary
        ...

        ## Description
        ...
        ---
        While generating the report, consider the following:
        - The harness code is not the root cause and it serves as an entry point for the fuzzer to test the codebase.
        - Make sure to cover all important aspects of the issue and the patch attempt.
        - Do not speculate on the causes of the issue or provide any solutions.
        """
    )
    added_issue_template = inspect.cleandoc(
        """\
        ## Initial Issue Log
        Here is the initial issue log of the codebase:
        <issue>
        {issue}
        </issue>
        """
    )
    added_issue_after_patch_template = inspect.cleandoc(
        """\
        ## Additional Issue Log
        Here is the additional issue log after the last patch attempt:
        <additional_issue>
        {additional_issue}
        </additional_issue>

        ## Notes
        - All patch attempts are applied to the initial codebase. Reapply the the previous patches if needed.
        - If multiple different issues are present, all the issues must be resolved in a single patch attempt.
        - Try different approaches to resolve the issues.
        """
    )

    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(llm)
        self.max_n_log_chars = 16000

    def __call__(self, state: EvaluationTeamState) -> dict[str, Any]:
        if state.issue == "":
            raise ValueError("Issue must be provided")

        if state.evaluation_report == "" and state.diff == "":
            user_content = self.user_prompt_issue_only.format(issue=state.issue)
            is_first_evaluation = True
        else:
            user_content = self.user_prompt_with_previous_report.format(
                issue=state.issue,
                patch_attempt=state.diff,
                patch_result=state.patch_result,
            )
            is_first_evaluation = False

        state.messages = [
            SystemMessage(
                content=self.system_prompt,
            ),
            HumanMessage(
                content=user_content,
            ),
        ]
        state.messages.append(self.llm.invoke(state.messages))
        evaluation_report = self._check_content_is_str(state.messages[-1].content)  # type: ignore
        evaluation_report += "\n\n" + self.added_issue_template.format(
            issue=state.issue
        )
        if not is_first_evaluation:
            evaluation_report += "\n\n" + self.added_issue_after_patch_template.format(
                additional_issue=state.patch_result,
            )
        return {"evaluation_report": evaluation_report}
