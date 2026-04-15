import inspect
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from crete.atoms.action import (
    Action,
    CompilableDiffAction,
    SoundDiffAction,
    UncompilableDiffAction,
    UnknownErrorAction,
    VulnerableDiffAction,
)
from crete.atoms.detection import Detection
from crete.commons.crash_analysis import get_crash_stacks
from crete.commons.interaction import CommandInteractionError, TimeoutExpired
from crete.agent.context import AgentContext
from crete.atoms.detection import AIxCCChallengeDeltaMode
from crete.prism.states.common_state import PatchStatus
from crete.prism.states.evaluation_team_state import (
    EvaluationTeamState,
)
from crete.prism.teams.base_agent import BaseAgent
from crete.analyzer.jvm_stackoverflow import JVMStackOverflowStacktraceAnalyzer
from crete.analyzer.jvm_timeout import JVMTimeoutStacktraceAnalyzer
from crete.environment.exceptions import (
    ChallengePoVFoundError,
    ChallengeTestFailedError,
)
from pathlib import Path

class Evaluator(BaseAgent):
    language_map = {
        "c": "c",
        "c++": "cpp",
        "cpp": "cpp",
        "java": "java",
        "jvm": "java",
    }
    abbreviated_log_template: str = "{log}\n... [TRUNCATED]"
    internal_tests_failure_prompt = inspect.cleandoc(
        """
        Your patch has fixed the issue, but the tests are failing.
        Here are some possible reasons for the failure:
        - The patch fixed the issue, but altered the original code's behavior.
        - The patch was too aggressive and removed or altered necessary code.
        - The patch did not adhere to the original code's logic.
        - There exists better patch locations that can fix the issue without introducing regressions.
        """
    )
    empty_patch_prompt = inspect.cleandoc(
        """
        The patch did not apply, here are some possible reasons:
        - Trying to patch a file that does not exist in the repository.
        - Trying to patch fuzzer or harness related file. (These files are used to test the validity of the patch.)
        - Incorrect patch format. (Focus on the patch formatting instructions.)
        """
    )
    sarif_location_template = inspect.cleandoc(
        """
        <sarif_report>
        Detected a potential issue in the codebase:
        - Physical Location: {physical_location}
        - Logical Locations: {logical_locations}
        - Message: {message}
        - Kind: {kind}
        - Severity: {severity}
        </sarif_report>
        """
    )
    sarif_no_value_prompt = "Not available"
    related_diff_template = inspect.cleandoc(
        """
        <related_diff>
        The following diff may be related to the initial issue:
        ```
        {related_diff}
        ```
        If the diff is not related to the current issue, you can ignore it.
        </related_diff>
        """
    )
    diff_header_regex = re.compile(r"^(?:--- |\+\+\+ |@@).*$", re.MULTILINE)

    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(llm)
        self.context: AgentContext | None = None
        self.detection: Detection | None = None
        self.max_n_log_chars = 16000

    def set_context_and_detection(
        self, context: AgentContext, detection: Detection
    ) -> None:
        self.context = context
        self.detection = detection

    def _environment_run_pov(self) -> Action:
        if self.context is None or self.detection is None:
            raise ValueError("Context and detection must be provided")
        environment = self.context["pool"].restore()
        try:
            if not self.detection.blobs:
                return SoundDiffAction(diff=b"")
            blob = self.detection.blobs[0]
            response_dir = self.context["output_directory"] / "prism-initial-eval"
            response_dir.mkdir(parents=True, exist_ok=True)
            rebuild_id = None
            environment.run_pov(
                pov_path=_write_blob_to_temp_file(blob.blob),
                harness_name=blob.harness_name,
                rebuild_id=rebuild_id,
                response_dir=response_dir,
            )
        except ChallengePoVFoundError as e:
            return VulnerableDiffAction(diff=b"", stdout=e.stdout, stderr=e.stderr)
        except Exception as e:  # pylint: disable=broad-except
            return UnknownErrorAction(error=e)
        return SoundDiffAction(diff=b"")

    def __call__(self, state: EvaluationTeamState) -> dict[str, Any]:
        if self.context is None or self.detection is None:
            raise ValueError("Context and detection must be provided")

        if state.patch_status == PatchStatus.INITIALIZED:
            # NOTE: Handle cases that may not have restored.
            self.context["pool"].restore()

            action = self._environment_run_pov()
            # Sarif only case
            if (
                not isinstance(action, VulnerableDiffAction)
                and self.detection.sarif_report is not None
            ):
                action = VulnerableDiffAction(diff=b"", stdout=b"", stderr=b"")
        elif state.diff == "":
            action = UncompilableDiffAction(
                variant="uncompilable",
                diff=b"",
                stdout=b"",
                stderr=b"Patch not applicable due to empty diff",
            )
        else:
            action = self.context["evaluator"].evaluate(
                self.context, bytes(state.diff, "utf-8"), self.detection
            )
            self.context["pool"].restore()

        patch_status = PatchStatus.from_action(action)
        repo_lang = self.language_map[self.detection.language]
        patch_result = self._add_sarif_logs(
            self._filter_action_log(
                (
                    getattr(action, "stdout", b"") + getattr(action, "stderr", b"")
                ).decode(errors="replace"),
                repo_lang,
                patch_status,
            )
        )
        if state.issue == "":
            state.issue = self._add_related_diff(patch_result)

        tests_log = ""
        if patch_status == PatchStatus.SOUND:
            # TODO: Add test logs in the issue. (Currently too long to naively add)
            tests_log, patch_status = self._environment_run_tests(
                state.diff, patch_status
            )
            self.context["pool"].restore()
            if patch_status == PatchStatus.TESTS_FAILED:
                patch_result = self.internal_tests_failure_prompt
        elif patch_status == PatchStatus.COMPILABLE:
            if "no tests" in patch_result:
                # NOTE: Some issues have "no tests" this is a hard-coded workaround to avoid such cases.
                patch_status = PatchStatus.SOUND
            else:
                # NOTE: Otherwise, COMPILABLE is the test failure case.
                patch_status = PatchStatus.TESTS_FAILED
                patch_result = self.internal_tests_failure_prompt
        elif patch_status == PatchStatus.UNCOMPILABLE and state.diff == "":
            # If the patch is empty, provide a more informative message
            patch_result = self.empty_patch_prompt

        if "logger" in self.context:
            log = (
                f"===Evaluator Result===\n{patch_status}\n"
                + f"===Evaluator Diff===\n{state.diff}\n"
                + f"===Patch Result===\n{patch_result}\n"
                + f"===Issue===\n{state.issue}\n"
                + f"===Evaluation Report===\n{state.evaluation_report}\n"
                + f"===Analysis Report===\n{state.analysis_report}\n"
            )
            if patch_status == PatchStatus.SOUND:
                if tests_log == "Tests skipped. No tests found.":
                    log += "===Tests Result===\nNo tests found\n"
                elif tests_log == "Command interaction error while testing.":
                    log += (
                        "===Tests Result===\nCommand interaction error while testing\n"
                    )
                else:
                    log += "===Tests Result===\nPassed\n"
            elif patch_status == PatchStatus.TESTS_FAILED:
                log += "===Tests Result===\nFailed\n"
            log += "===Evaluation End==="
            self.context["logger"].info(log)
        return {
            "patch_status": patch_status,
            "issue": state.issue,
            "patch_result": patch_result,
            "diff": state.diff,
            "repo_lang": repo_lang,
            "tests_log": tests_log,
        }

    def _filter_vulnerable_log(self, action_log: str, lang: str) -> str:
        if len(action_log) <= self.max_n_log_chars:
            return action_log
        vulnerability_prefix = None
        vulnerability_postfix = None
        if lang in ("c", "cpp"):
            vulnerability_prefix = "====="
            vulnerability_postfix = "==ABORTING"
        elif lang == "java":
            vulnerability_prefix = "== Java Exception:"
            vulnerability_postfix = "== libFuzzer crashing input =="

        if vulnerability_prefix is not None and vulnerability_prefix in action_log:
            action_log = (
                vulnerability_prefix
                + action_log.split(vulnerability_prefix, maxsplit=1)[1]
            )

        if vulnerability_postfix is not None and vulnerability_postfix in action_log:
            action_log = (
                action_log.rsplit(vulnerability_postfix, maxsplit=1)[0]
                + vulnerability_postfix
            )
            # Reduce token usage by removing shadow bytes log
            shadow_bytes_text = "Shadow bytes around the buggy address:"
            if shadow_bytes_text in action_log:
                action_log = action_log.split(shadow_bytes_text, maxsplit=1)[0]

        if len(action_log) > self.max_n_log_chars:
            action_log = self.abbreviated_log_template.format(
                log=action_log[: self.max_n_log_chars]
            )
        return action_log

    def _filter_uncompilable_log(self, action_log: str, lang: str) -> str:
        if len(action_log) <= self.max_n_log_chars:
            return action_log
        build_failure_prefix = None
        build_failure_postfix = None
        if lang in ("c", "cpp"):
            build_failure_prefix = "error:"
            build_failure_postfix = "errors generated."
        elif lang == "java":
            build_failure_prefix = "ERROR"
            build_failure_postfix = "For more information about the errors"

        if build_failure_prefix is not None and build_failure_prefix in action_log:
            log_before_prefix, log_after_prefix = action_log.split(
                build_failure_prefix, maxsplit=1
            )
            # Add the whole line before the prefix
            prefix = build_failure_prefix
            if "\n" in log_before_prefix:
                prefix = log_before_prefix.rsplit("\n", maxsplit=1)[1] + prefix
            action_log = prefix + log_after_prefix
        if build_failure_postfix is not None and build_failure_postfix in action_log:
            action_log = (
                action_log.rsplit(build_failure_postfix, maxsplit=1)[0]
                + build_failure_postfix
            )
        if len(action_log) > self.max_n_log_chars:
            action_log = self.abbreviated_log_template.format(
                log=action_log[: self.max_n_log_chars]
            )
        return action_log

    def _filter_java_timeout_log(self, action_log: str) -> str:
        # Filter out timeout logs without helpful crash stacks.
        if "ERROR: libFuzzer: timeout" not in action_log:
            return action_log

        if self.context is None or self.detection is None:
            return action_log

        try:
            if get_crash_stacks(self.context, self.detection) is not None:
                return action_log

            # Reoccuring trace are abbreviated
            timeout_stack_bytes = JVMTimeoutStacktraceAnalyzer().analyze(
                self.context, self.detection
            )
            if timeout_stack_bytes is None:
                return action_log
        except Exception as e:
            if "logger" in self.context:
                self.context["logger"].info(
                    f"Evaluator: JVMTimeoutStacktraceAnalyzer failed: {e}",
                    exc_info=True,
                )
            return action_log

        timeout_stack_str = timeout_stack_bytes.decode(errors="replace")
        return "ERROR: libFuzzer: timeout\n" + timeout_stack_str

    def _filter_java_stackoverflow_log(self, action_log: str) -> str:
        # Filter out timeout logs without helpful crash stacks.
        if "FuzzerSecurityIssueLow: Stack overflow" not in action_log:
            return action_log

        if self.context is None or self.detection is None:
            return action_log

        try:
            if get_crash_stacks(self.context, self.detection) is not None:
                return action_log

            overflow_stack = JVMStackOverflowStacktraceAnalyzer().analyze(
                self.context, self.detection
            )
            if overflow_stack is None:
                return action_log
        except Exception as e:
            if "logger" in self.context:
                self.context["logger"].info(
                    f"Evaluator: JVMStackOverflowStacktraceAnalyzer failed: {e}",
                    exc_info=True,
                )
            return action_log

        action_log = "FuzzerSecurityIssue: Stack overflow\n" + overflow_stack
        # Abbreviate the log if it is too long
        if len(action_log) > self.max_n_log_chars:
            visible_len = self.max_n_log_chars // 2
            action_log = (
                action_log[:visible_len] + "\n...\n" + action_log[-visible_len:]
            )
        return action_log

    def _filter_action_log(
        self, action_log: str, lang: str, patch_status: PatchStatus
    ) -> str:
        if action_log == "":
            return action_log
        # This is a hard coded filtering for the oss-fuzz logs
        if patch_status == PatchStatus.VULNERABLE:
            if lang == "java":
                action_log = self._filter_java_timeout_log(action_log)
                action_log = self._filter_java_stackoverflow_log(action_log)

            action_log = self._filter_vulnerable_log(action_log, lang)
        elif patch_status == PatchStatus.UNCOMPILABLE:
            action_log = self._filter_uncompilable_log(action_log, lang)

        if len(action_log) > self.max_n_log_chars and not action_log.endswith(
            "[TRUNCATED]"
        ):
            action_log = self.abbreviated_log_template.format(
                log=action_log[: self.max_n_log_chars]
            )

        # Prevent crash logs instructing LLM to allow network connections (#1007)
        action_log = action_log.replace(
            "If the fuzz test is expected to perform network connections,"
            " call com.code_intelligence.jazzer.api.BugDetectors#allowNetworkConnections"
            " at the beginning of your fuzz test and optionally provide a predicate matching the expected hosts.",
            "",
        )
        return action_log

    def _add_sarif_logs(self, action_log: str) -> str:
        if self.context is None or self.detection is None:
            return action_log
        sarif_report = self.detection.sarif_report
        if sarif_report is None:
            return action_log
        if len(sarif_report.runs) == 0:
            return action_log

        sarif_logs: list[str] = []
        for sarif_run in sarif_report.runs:
            results = sarif_run.get("results")
            if not isinstance(results, list) or len(results) == 0:
                continue
            for result in results:
                formatted_sarif_result = self._format_sarif_result(result)
                if formatted_sarif_result == "":
                    continue
                else:
                    sarif_logs.append(formatted_sarif_result)

        return action_log + "\n" + "\n".join(sarif_logs)

    def _format_sarif_result(self, sarif_result: Any) -> str:
        if not isinstance(sarif_result, dict):
            return ""
        locations = sarif_result.get("locations")
        if not isinstance(locations, list) or len(locations) == 0:
            return ""

        formatted_sarif_locations: list[str] = []
        for location in locations:
            if not isinstance(location, dict):
                continue
            physical_location_prompt = self.sarif_no_value_prompt
            logical_locations_prompt = self.sarif_no_value_prompt
            message_prompt = self.sarif_no_value_prompt
            kind_prompt = self.sarif_no_value_prompt
            severity_prompt = self.sarif_no_value_prompt

            # Physical Location
            physical_location = location.get("physicalLocation")
            if isinstance(physical_location, dict):
                artifact_location = physical_location.get("artifactLocation")
                region = physical_location.get("region")
                file_path = None
                if isinstance(artifact_location, dict):
                    file_path = artifact_location.get("uri")
                line_start = None
                line_end = None
                if isinstance(region, dict):
                    line_start = region.get("startLine")
                    line_end = region.get("endLine")
                if isinstance(file_path, str) and isinstance(line_start, int) and isinstance(line_end, int):
                    physical_location_prompt = f"{file_path}:{line_start}-{line_end}"
                elif isinstance(file_path, str):
                    physical_location_prompt = file_path

            # Logical Location
            logical_locations_raw = location.get("logicalLocations")
            if isinstance(logical_locations_raw, list) and len(logical_locations_raw) > 0:
                logical_locations: list[str] = []
                for logical_location in logical_locations_raw:
                    if not isinstance(logical_location, dict):
                        continue
                    loc_prompt = ""
                    if isinstance(logical_location.get("name"), str):
                        loc_prompt += logical_location["name"]
                    if isinstance(logical_location.get("kind"), str):
                        loc_prompt += f"({logical_location['kind']})"
                    if loc_prompt != "":
                        logical_locations.append(loc_prompt)
                if len(logical_locations) > 0:
                    logical_locations_prompt = ", ".join(logical_locations)

            # Message
            message = sarif_result.get("message")
            if isinstance(message, dict) and isinstance(message.get("text"), str):
                message_prompt = message["text"]

            # Kind
            if sarif_result.get("kind") is not None:
                kind_prompt = str(sarif_result["kind"])

            # Severity
            if sarif_result.get("level") is not None:
                severity_prompt = str(sarif_result["level"])

            formatted_sarif_locations.append(
                self.sarif_location_template.format(
                    physical_location=physical_location_prompt,
                    logical_locations=logical_locations_prompt,
                    message=message_prompt,
                    kind=kind_prompt,
                    severity=severity_prompt,
                )
            )
        return "\n".join(formatted_sarif_locations)

    def _add_related_diff(self, issue: str) -> str:
        if self.context is None or self.detection is None:
            return issue
        if not isinstance(self.detection.mode, AIxCCChallengeDeltaMode):
            return issue
        # try:
        #     delta_diffs = get_all_diff(self.context, self.detection)
        # except Exception as e:
        #     if "logger" in self.context:
        #         self.context["logger"].info(
        #             f"Prism evaluator: get_all_diff failed: {e}",
        #             exc_info=True,
        #         )
        #     return issue

        # if delta_diffs is None or len(delta_diffs) == 0:
        #     return issue

        # # delta_diffs is a list of tuples (commit, diff)
        # delta_diffs_concat = "\n".join(d[1] for d in delta_diffs)

        # # diffs containing "aixcc" are filtered out
        # diffs = delta_diffs_concat.split("diff --git")
        # valid_diffs = [d for d in diffs if "aixcc" not in d and d.strip() != ""]

        # if len(valid_diffs) == 0:
        #     return issue

        # related_diff = "\n".join(valid_diffs)

        # @NOTE: comply with the OSS-CRS interface
        ref_diff_path = Path("/work/hints/ref.diff")
        if not ref_diff_path.exists():
            self.context["logger"].warning(f'delta mode, but `ref.diff` is not found in "{ref_diff_path}"')
            return issue
        related_diff = ref_diff_path.read_text(encoding="utf-8", errors="ignore")

        if len(related_diff) > self.max_n_log_chars:
            related_diff_not_abbr_part = related_diff[: self.max_n_log_chars]
            diff_headers = self.diff_header_regex.findall(
                related_diff[self.max_n_log_chars :]
            )
            diff_headers = [dh for dh in diff_headers if isinstance(dh, str)]
            if len(diff_headers) == 0:
                related_diff = related_diff_not_abbr_part + "\n..."
            else:
                diff_headers_with_abbr: list[str] = []
                for header in diff_headers:
                    if header.startswith("---"):
                        diff_headers_with_abbr.append("...")
                    diff_headers_with_abbr.append(header)
                related_diff_abbr_part = "\n".join(diff_headers_with_abbr)
                if len(related_diff_abbr_part) > self.max_n_log_chars:
                    related_diff_abbr_part = related_diff_abbr_part[
                        : self.max_n_log_chars
                    ]
                related_diff = (
                    related_diff_not_abbr_part + "\n" + related_diff_abbr_part + "\n..."
                )

        related_diff = related_diff.strip()
        if related_diff == "":
            return issue

        return (
            issue + "\n" + self.related_diff_template.format(related_diff=related_diff)
        )

    def _environment_run_tests(
        self, diff: str, current_patch_status: PatchStatus
    ) -> tuple[str, PatchStatus]:
        if self.context is None or self.detection is None:
            raise ValueError("Context and detection must be provided")
        if current_patch_status != PatchStatus.SOUND:
            return "Tests skipped. Provide a sound patch.", current_patch_status

        try:
            if not self.context["pool"].internal_test_exists():
                return "Tests skipped. No tests found.", current_patch_status

            environment = self.context["pool"].restore()
            response_dir = self.context["output_directory"] / "prism-test-eval"
            response_dir.mkdir(parents=True, exist_ok=True)
            environment.patch(diff.encode("utf-8", errors="replace"), response_dir)
            stdout, _ = environment.run_tests(diff.encode("utf-8", errors="replace"), response_dir)
            tests_log = stdout
        except ChallengeTestFailedError as e:
            tests_log = (e.stdout + e.stderr).decode(errors="replace")
            current_patch_status = PatchStatus.TESTS_FAILED
        except CommandInteractionError:
            tests_log = "Command interaction error while testing."
            # NOTE: Since we cannot decide if tests failed or not, we assume the patch is sound.
            current_patch_status = PatchStatus.SOUND
        except TimeoutExpired as e:
            tests_log = (e.stdout + e.stderr).decode(errors="replace")
            current_patch_status = PatchStatus.TESTS_FAILED
        return tests_log, current_patch_status


def _write_blob_to_temp_file(blob: bytes) -> Path:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pov", delete=False) as pov_file:
        pov_file.write(blob)
        return Path(pov_file.name)
