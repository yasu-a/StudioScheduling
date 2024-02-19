import abc
import functools
import typing

import numpy as np


class AgentBehaviorMixin(metaclass=abc.ABCMeta):
    @classmethod
    def _agent_shape(cls) -> tuple[int, ...]:
        raise NotImplementedError()

    @classmethod
    def _random_initial(cls) -> np.ndarray:  # ndarray with shape cls._agent_shape
        raise NotImplementedError()

    @classmethod
    def _score(cls, array: np.ndarray) -> float:
        raise NotImplementedError()

    @classmethod
    def _is_better(cls, current: float, x: float) -> bool:
        raise NotImplementedError()

    @classmethod
    def _neighbours(cls, array: np.ndarray, k: int | None) -> np.ndarray:
        raise NotImplementedError()


class AgentBase(AgentBehaviorMixin, metaclass=abc.ABCMeta):
    def __init__(self, array: np.ndarray, do_copy=True):
        if array is None:
            self._array = None  # agent with worst score
        else:
            self._array = np.copy(array)
            if do_copy:
                self._array = np.copy(self._array)
            self._array.setflags(write=False)

    def __repr__(self):
        return f'Agent<{self._array!r}>'

    @classmethod
    def create_random(cls):
        return cls(cls._random_initial(), do_copy=False)

    @functools.cached_property
    def score(self):
        return self._score(self._array)

    def clone(self) -> 'AgentBase':
        return type(self)(self._array)

    def neighbours(self, k) -> list['AgentBase']:
        return [
            type(self)(neighbour_array)
            for neighbour_array in self._neighbours(self._array, k)
        ]

    def neighbour(self) -> 'AgentBase':
        return self.neighbours(k=1)[0]

    def far_neighbour(self, k: int) -> 'AgentBase':
        a = self
        for _ in range(k):
            a = a.neighbour()
        return a

    def is_better_than(self, a: 'AgentBase'):
        return self._is_better(self.score, a.score)

    def stochastic_best_neighbour(self, k: int):
        current = None
        for neighbour in self.neighbours(k):
            if current is None or neighbour.is_better_than(current):
                current = neighbour
        return current

    def stochastic_hill_climbing(self, k: int, b: int):
        current = self
        no_improvement_count = 0
        while True:
            best_neighbour = current.stochastic_best_neighbour(k)
            if best_neighbour.is_better_than(current):
                current = best_neighbour
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= b:
                    break
        return current

    def multiple_stochastic_hill_climbing(self, k: int, b: int, n: int):
        better = None
        for _ in range(n):
            current = self.stochastic_hill_climbing(k, b)
            if current.is_better_than(self):
                better = current
        return better
