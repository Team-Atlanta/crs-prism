from enum import Enum, auto

from pydantic import BaseModel

from crete.atoms.action import (
    Action,
    CompilableDiffAction,
    NoPatchAction,
    SoundDiffAction,
    UncompilableDiffAction,
    UnknownErrorAction,
    VulnerableDiffAction,
    WrongDiffAction,
)
from crete.state.patch_state import (
    CodeSnippet,
)


class PatchStatus(Enum):
    # TODO: Group this with the one in multi_retrieval.
    INITIALIZED = auto()
    UNCOMPILABLE = auto()
    COMPILABLE = auto()
    VULNERABLE = auto()
    WRONG = auto()
    TESTS_FAILED = auto()
    SOUND = auto()
    UNKNOWN = auto()

    @classmethod
    def from_action(cls, action: Action) -> "PatchStatus":
        patch_status = cls.INITIALIZED
        match action:
            case UncompilableDiffAction():
                patch_status = cls.UNCOMPILABLE
            case CompilableDiffAction():
                patch_status = cls.COMPILABLE
            case VulnerableDiffAction():
                patch_status = cls.VULNERABLE
            case WrongDiffAction():
                patch_status = cls.WRONG
            case SoundDiffAction():
                patch_status = cls.SOUND
            case UnknownErrorAction():
                patch_status = cls.UNKNOWN
            case NoPatchAction():
                patch_status = cls.UNCOMPILABLE
            case _:
                raise ValueError(f"Unknown action type: {type(action)}")
        return patch_status


class CommonState(BaseModel):
    patch_status: PatchStatus = PatchStatus.INITIALIZED
    applied_patches: list[CodeSnippet] = []
    analysis_report: str = ""
    evaluation_report: str = ""
    relevant_code_snippets: str = ""
    diff: str = ""
    issue: str = ""
    repo_path: str = ""

    @classmethod
    def from_common_state(cls, common_state: "CommonState") -> "CommonState":
        return cls(
            patch_status=common_state.patch_status,
            applied_patches=common_state.applied_patches,
            analysis_report=common_state.analysis_report,
            evaluation_report=common_state.evaluation_report,
            relevant_code_snippets=common_state.relevant_code_snippets,
            diff=common_state.diff,
            issue=common_state.issue,
            repo_path=common_state.repo_path,
        )
