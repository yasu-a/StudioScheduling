import abc
import functools

import numpy as np
from tqdm import tqdm


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
    def _neighbours(cls, array: np.ndarray, k: int | None) -> np.ndarray:
        """Returns k of random neighbours"""
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

    def __eq__(self, other):
        assert isinstance(other, AgentBase)
        return np.all(self._array == other._array)

    def __hash__(self):
        return hash(tuple(self._array.flatten()))

    @classmethod
    def create_random_agent_set(cls, n) -> 'AgentSet':
        return AgentSet([
            cls(cls._random_initial(), do_copy=False)
            for _ in range(n)
        ])

    @functools.cached_property
    def score(self):
        return self._score(self._array)

    def clone(self) -> 'AgentBase':
        return type(self)(self._array)

    def neighbours(self, k) -> 'AgentSet':
        return AgentSet([
            type(self)(neighbour_array)
            for neighbour_array in self._neighbours(self._array, k)
        ])

    def neighbour(self) -> 'AgentBase':
        return self.neighbours(k=1).one()

    def far_neighbour(self, k: int) -> 'AgentBase':
        a = self
        tabu = set()
        for _ in range(k):
            tabu_count = 0
            while True:
                a = a.neighbour()
                if a not in tabu:
                    break
                tabu_count += 1
                if tabu_count >= 6:
                    break
            tabu.add(a)
        return a

    def stochastic_best_neighbour(self, k: int):
        current = None
        for neighbour in self.neighbours(k):
            if current is None or neighbour.score > current.score:
                current = neighbour
        return current

    def stochastic_hill_climbing(self, k: int):
        current = self
        while True:
            best_neighbour = current.stochastic_best_neighbour(k)
            if not best_neighbour.score > current.score:
                break
            current = best_neighbour
        return current

    def description(self) -> str:
        raise NotImplementedError()


class AgentSet:
    def __init__(self, agents: list[AgentBase]):
        self._agents = agents

    def __len__(self):
        return len(self._agents)

    def one(self) -> AgentBase:
        assert len(self) == 1
        return self._agents[0]

    def __iter__(self):
        yield from self._agents

    @functools.cached_property
    def scores(self):
        sa = np.array([a.score for a in self._agents])
        sa.setflags(write=False)
        return sa

    @functools.cached_property
    def argsort(self):
        a = np.argsort(self.scores)
        a.setflags(write=False)
        return a

    def best_one(self) -> 'AgentBase':
        return self._agents[self.argsort[-1]]

    def choose_best(self, n):
        if n <= 0:
            return []
        return [self._agents[i] for i in self.argsort[-n:]]

    def choose_worst(self, n):
        if n <= 0:
            return []
        return [self._agents[i] for i in self.argsort[:n]]

    def choose_weighted_random(self, n, gamma):
        s = self.scores
        w = (s - s.mean()) / s.std()
        w = np.clip(w, -2, 2) / 2 + 1 + 1e-6
        w **= gamma
        assert np.all(w > 0)
        w /= w.sum()
        idx = np.random.choice(np.arange(len(self)), size=n, p=w)
        return [self._agents[i] for i in idx]

    def apply(self, mapper):
        for i in range(len(self)):
            self._agents[i] = mapper(self._agents[i])


class SolverBase:
    @classmethod
    def _agent_class(cls) -> type[AgentBase]:
        raise NotImplementedError()

    def __init__(self, *, n_agents, n_max_iter):
        self._a_set: AgentSet = self._agent_class().create_random_agent_set(n=n_agents)
        self._best = self._a_set.best_one()
        self._n_max_iter = n_max_iter

    def reset_agents(self):
        raise NotImplementedError()

    def improve(self, a: AgentBase) -> AgentBase:
        raise NotImplementedError()

    def finished(self):
        raise NotImplementedError()

    def update_best(self, current_best: AgentBase) -> AgentBase:
        if current_best.score > self._best.score:
            return current_best
        return self._best

    def step(self, tqdm_bar=None):
        self._a_set.apply(self.improve)

        current_best = self._a_set.best_one()
        self._best = self.update_best(current_best)
        if tqdm_bar:
            tqdm_bar.set_description(f'{self._best.description()} {current_best.description()}')

        if self.finished():
            return True

        self.reset_agents()
        return False

    def solve(self):
        bar = tqdm(range(self._n_max_iter))
        for _ in bar:
            finished = self.step(tqdm_bar=bar)
            if finished:
                break
