import random
from typing import Any, List, Optional


class Experience:
    def __init__(self, observation: Any, action: Any, reward: float, next_observation: Optional[Any]):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation

    @property
    def is_terminal(self):
        return self.next_observation is None


class ReplayBuffer:
    def __init__(self, size: int = 100):
        self.size = size
        self._experiences: List[Experience] = []
        self._next_to_overwrite = 0

    def _is_full(self):
        return len(self._experiences) == self.size

    def add(self, experience: Experience):
        if self._is_full():
            self._experiences[self._next_to_overwrite] = experience
            self._next_to_overwrite = (self._next_to_overwrite + 1) % self.size
        else:
            self._experiences.append(experience)

    def sample(self, num: int):
        return random.choices(self._experiences, k=num)

    def __len__(self):
        return len(self._experiences)
