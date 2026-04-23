from dataclasses import dataclass
from time import perf_counter


@dataclass
class BudgetTracker:
    forward_passes: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0

    def start(self) -> None:
        self.started_at = perf_counter()

    def stop(self) -> None:
        self.ended_at = perf_counter()

    def add_forward(self, count: int = 1) -> None:
        self.forward_passes += count

    @property
    def wall_clock_s(self) -> float:
        if self.ended_at <= self.started_at:
            return 0.0
        return self.ended_at - self.started_at
