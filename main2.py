import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from project_data import ProjectData
from solver import AgentBase

if __name__ == '__main__':
    json_name = 'test.json'
    proj_data = ProjectData.from_cache(json_name)


class SolverAgent(AgentBase):
    @classmethod
    def _agent_shape(cls) -> tuple[int, ...]:
        return proj_data.n_timespan,

    @classmethod
    def _random_initial(cls) -> np.ndarray:
        return np.random.randint(
            proj_data.n_band_set,
            size=cls._agent_shape()
        )

    @classmethod
    def _score(cls, array: np.ndarray) -> float:  # minimize, minimum is 0
        band_occ_count = proj_data.table_band_occurrence_count(array)
        band_occ_error = np.abs(band_occ_count - proj_data.n_max_band_occurrence)
        band_occ_error_score = band_occ_error.mean() / proj_data.n_max_band_occurrence
        return band_occ_error_score

    @classmethod
    def _is_better(cls, current: float, x: float) -> bool:
        return current < x  # minimize: smaller is better

    @classmethod
    def _neighbours(cls, array: np.ndarray, k: int | None) -> np.ndarray:
        k = k or 1
        j = np.random.randint(cls._agent_shape()[0], size=k)
        neighbours = np.repeat(array[None, :], axis=0, repeats=k)
        for i in range(k):
            a = proj_data.get_neighbours(neighbours[i, j[i]])
            neighbours[i, j[i]] = np.random.choice(a)
        return neighbours

    def finished(self):
        return proj_data.band_occupation_rate(self._array) > 0.999

    def description(self) -> str:
        band_occ_count = proj_data.table_band_occurrence_count(self._array)
        occ_rate = proj_data.band_occupation_rate(self._array)
        return f'{self.score:.3f} {"".join(map(str, band_occ_count))} {occ_rate * 100:.0f}%'


class Solver:
    def __init__(self, n_agents=128):
        self.__n_agents = n_agents
        self.__agents = [
            SolverAgent.create_random()
            for _ in range(self.__n_agents)
        ]

    def scores(self):
        return np.array([a.score for a in self.__agents])

    def best(self):
        return self.__agents[self.scores().argmax()]

    def __choose_best(self, n):
        if n == 0:
            return []
        s = self.scores()
        idx = s.argsort()[-n:]
        return [self.__agents[i].clone() for i in idx]

    def __choose_worst(self, n):
        if n == 0:
            return []
        s = self.scores()
        idx = s.argsort()[:n]
        return [self.__agents[i].clone() for i in idx]

    def __choose_middle(self, n):
        if n == 0:
            return []
        s = self.scores()
        w = (s - s.mean()) / s.std()
        w = np.clip(w + 2, 0, 4) / 4
        w **= 16
        w /= w.sum()
        idx = np.random.choice(np.arange(len(self.__agents)), size=n, p=w)
        return [self.__agents[i].clone() for i in idx]

    def reset_agents(self, n_best=0, n_worst=0):
        new_agents = []
        for a in self.__choose_best(n_best):
            new_agents.append(a.neighbour())
        for a in self.__choose_worst(n_worst):
            new_agents.append(a.far_neighbour(k=4))
        for a in self.__choose_middle(self.__n_agents - n_best - n_worst):
            new_agents.append(a.neighbour())

        assert len(new_agents) == self.__n_agents
        self.__agents = new_agents

    def solve(self):
        bar = tqdm(range(256))
        best = None
        for k in bar:
            for i in range(self.__n_agents):
                a = self.__agents[i].stochastic_hill_climbing(k=64, b=4)
                if a is not None:
                    self.__agents[i] = a

            current_best = self.best()
            if best is None or current_best.is_better_than(best):
                best = current_best
            bar.set_description(f'{best.description()} {current_best.description()}')

            if best.finished():
                break

            self.reset_agents()


def test_agent_hill_climbing():
    plt.figure(figsize=(10, 5))
    k, b, n = 16, 4, 4

    def job():
        a = SolverAgent.create_random()
        history = [a.score]
        for _ in range(8):
            r = a.multiple_stochastic_hill_climbing(k=k, b=b, n=n)
            if r is None:
                for _ in range(2):
                    a = a.neighbour()
            else:
                a = r.neighbour()
            history.append(a.score)
        history = np.array(history)
        return history

    for history in joblib.Parallel(n_jobs=-1, verbose=2)(joblib.delayed(job)() for _ in range(64)):
        plt.plot(history, alpha=0.5)
    plt.title(f'{k=}, {b=}, {n=}')
    plt.ylim(0, 0.5)
    plt.show()


def main():
    solver = Solver()
    solver.solve()


if __name__ == '__main__':
    main()
