from buffers import Evictor


class FIFOEvictor(Evictor):
    def evict(self, *args, **kwargs) -> int:
        return 0
