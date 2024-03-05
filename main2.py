import numpy as np

from project_data import ProjectData
from solver import AgentBase, AgentSet, SolverBase

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
    def _score(cls, array: np.ndarray) -> float:  # maximize, maximum is 0
        band_occ_count = proj_data.table_band_occurrence_count(array)
        band_occ_error = np.abs(band_occ_count - proj_data.n_max_band_occurrence)
        band_occ_error_score = band_occ_error.mean() / proj_data.n_max_band_occurrence
        return -band_occ_error_score

    @classmethod
    def _neighbours(cls, array: np.ndarray, k: int | None) -> np.ndarray:
        k = k or 1
        j = np.random.randint(cls._agent_shape()[0], size=k)
        neighbours = np.repeat(array[None, :], axis=0, repeats=k)
        for i in range(k):
            a = proj_data.get_neighbours(neighbours[i, j[i]])
            neighbours[i, j[i]] = np.random.choice(a)
        return neighbours

    def band_occupation_rate(self):
        return proj_data.band_occupation_rate(self._array)

    def description(self) -> str:
        band_occ_count = proj_data.table_band_occurrence_count(self._array)
        occ_rate = self.band_occupation_rate()
        return f'{self.score:.3f} {"".join(map(str, band_occ_count))} {occ_rate * 100:.0f}%'


class Solver(SolverBase):
    @classmethod
    def _agent_class(cls) -> type[AgentBase]:
        return SolverAgent

    def __init__(self, *, n_agents, n_max_iter, n_best=0, n_worst=0):
        super().__init__(n_agents=n_agents, n_max_iter=n_max_iter)

        self._n_best = n_best
        self._n_worst = n_worst

    def improve(self, a: AgentBase) -> AgentBase:
        return self._agent_class().stochastic_hill_climbing(a, k=16)

    def reset_agents(self):
        new_agents = []
        for a in self._a_set.choose_best(n=self._n_best):
            new_agents.append(a.far_neighbour(k=1))
        for a in self._a_set.choose_worst(n=self._n_worst):
            new_agents.append(a.far_neighbour(k=4))
        for a in self._a_set.choose_weighted_random(
                n=len(self._a_set) - self._n_best - self._n_worst,
                gamma=16
        ):
            new_agents.append(a.far_neighbour(k=2))

        assert len(new_agents) == len(self._a_set)
        self._a_set = AgentSet([a.clone() for a in new_agents])

    def finished(self):
        assert isinstance(self._best, SolverAgent)
        return self._best.band_occupation_rate() > 0.999


# class Solver2(SolverBase):
#     def __init__(self, *, n_agents, n_best=0, n_worst=0):
#         super().__init__(n_agents=n_agents)
#
#         self._n_best = n_best
#         self._n_worst = n_worst
#
#     def reset_agents(self):
#         from sklearn.cluster import HDBSCAN, KMeans
#         pass


def main():
    solver = Solver(
        n_agents=128,
        n_max_iter=512,
        n_best=32,
        n_worst=32,
    )
    solver.solve()


if __name__ == '__main__':
    main()
