from abc import ABC, abstractmethod


class Evictor(ABC):
    def __call__(self, *args, **kwargs) -> int:
        return self.evict(*args, **kwargs)

    @abstractmethod
    def evict(self, *args, **kwargs) -> int:
        """Return the index of the episode to evict."""
        raise NotImplementedError
