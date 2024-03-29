from dataclasses import dataclass, field
from buffers.dataset.fields import Fields
from buffers.dataset.verification import VerificationMode
from typing import List, Optional


@dataclass
class EpisodeDatasetInfo:
    fields: Optional[Fields] = None
    verification_mode: VerificationMode = VerificationMode.BASIC_CHECKS
    num_episodes: int = 0
    num_timesteps: int = 0
    episode_lengths: List[int] = field(default_factory=list)
    is_full: bool = False

    @property
    def is_empty(self) -> bool:
        return self.num_episodes == 0
