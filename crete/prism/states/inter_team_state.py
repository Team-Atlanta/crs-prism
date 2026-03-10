from enum import Enum, auto

from crete.prism.states.common_state import CommonState


class TeamStatus(Enum):
    START = auto()
    ANALYZE = auto()
    PATCH = auto()
    EVALUATE = auto()
    END = auto()


class InterTeamState(CommonState):
    team_status: TeamStatus = TeamStatus.START
    n_evals: int = 0
