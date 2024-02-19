import collections
import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm


class Agent:
    @classmethod
    @property
    def size(cls):
        return N_TIMESPANS

    def __init__(self, a=None):
        if a is None:
            self.__a = np.random.choice(b_nodup_index, size=self.size, replace=True)
        else:
            self.__a = a
        self.__score = None

    @property
    def score(self) -> float:
        if self.__score is None:
            self.__score = table_band_occurrence_score(self.__a)
        return self.__score

    def neighbour(self) -> 'Agent':
        a = self.__a.copy()
        i = np.random.randint(self.size)
        a[i] = np.random.choice(get_neighbours(a[i]))
        return Agent(a)

    def stochastic_better(self, k) -> 'Agent':
        better_agent = None
        for _ in range(k):
            n = self.neighbour()
            if better_agent is None or n.score > better_agent.score:
                better_agent = n
        return better_agent

    def local_search(self, k=16, improvement_thresh=1e-6) -> 'Agent':
        best = self

        while True:
            better = best.stochastic_better(k)
            improvement = better.score - best.score
            if improvement > 0:
                best = better
            if improvement < improvement_thresh:
                break

        return best

    def far_neighbour(self, k=2) -> 'Agent':
        a = self
        for _ in range(k):
            a = a.neighbour()
        return a

    def __repr__(self):
        return f'Agent<score={self.score:.2f}, a={self.__a}>'

    def clone(self):
        return Agent(self.__a.copy())

    def band_occurrence_count(self):
        return table_band_occurrence_count(self.__a)

    def root_occupation_rate(self):
        band_occ = self.band_occurrence_count().min()
        return band_occ * len(s_band) / MAX_N_BANDS

    def description(self):
        oc = self.band_occurrence_count()
        desc = f'{self.score:.3f} {"".join(map(str, oc))} {oc.min()} ' \
               f'{self.root_occupation_rate() * 100:.0f}%'
        return desc

    def table(self):
        return [
            [m_band_to_name[b] for b in s_band[b_mask]]
            for b_mask in s_band_nodup_member[self.__a].astype(bool)
        ]

    @property
    def array(self):
        return self.__a.copy()

    def occurrence_pair_matrix(self):
        mat = np.zeros(shape=(len(s_band), len(s_band)), dtype=np.float32)

        sb = s_band_nodup_member[self.array]

        for b in s_band:
            ts_mask = sb[:, b].astype(bool)
            mat[b, :] = sb[ts_mask, :].sum(axis=0)
            mat[b, b] = 0

        return mat


class State:
    def __init__(self, agents: list['Agent']):
        self.__agents = agents
        self.__history = [collections.defaultdict(list) for _ in range(2)]

    @classmethod
    def random_init(cls, n_agents=512):
        return State(agents=[Agent() for _ in range(n_agents)])

    def local_search(self):
        # self.__agents = joblib.Parallel(n_jobs=4)(
        #     joblib.delayed(a.local_search)()
        #     for a in self.__agents
        # )
        for i in range(len(self.__agents)):
            self.__agents[i] = self.__agents[i].local_search()

    def scores(self):
        return np.array([a.score for a in self.__agents])

    def update_score_stat(self):
        s = self.scores()
        self.__history[0]['best'].append(s.max())
        avg_target = (np.percentile(s, q=25) <= s) & (s <= np.percentile(s, q=75))
        self.__history[0]['avg'].append(s[avg_target].mean())
        self.__history[0]['min'].append(s.min())

        features = np.stack([
            a.occurrence_pair_matrix().flatten()
            for a in self.__agents
        ])
        from scipy.spatial.distance import euclidean
        g = features.mean(axis=0)
        variance = np.mean([euclidean(a, g) for a in features])
        self.__history[1]['feature_variance'].append(variance)

    def __choose_best(self, n):
        s = self.scores()
        idx = s.argsort()[-n:]
        return [self.__agents[i].clone() for i in idx]

    def __choose_worst(self, n):
        s = self.scores()
        idx = s.argsort()[:n]
        return [self.__agents[i].clone() for i in idx]

    def __choose_middle(self, n):
        s = self.scores()
        w = (s - s.mean()) / s.std()
        w = np.clip(w + 2, 0, 4) / 4
        w **= 16
        w /= w.sum()
        idx = np.random.choice(np.arange(len(self.__agents)), size=n, p=w)
        return [self.__agents[i].clone() for i in idx]

    def reset_agents(self, n_best=4, n_worst=32, r_worst=0.2):
        new_agents = []
        for a in self.__choose_best(n_best):
            new_agents.append(a.far_neighbour(k=1))
        for a in self.__choose_worst(n_worst):
            a = a.far_neighbour(k=1)
            if r_worst:
                a = a.far_neighbour(k=4)
            new_agents.append(a)
        for a in self.__choose_middle(len(self.__agents) - n_best - n_worst):
            new_agents.append(a.far_neighbour())

        assert len(self.__agents) == len(new_agents) and type(self.__agents) is type(new_agents)
        self.__agents = new_agents

    def best(self):
        return self.__agents[self.scores().argmax()]

    def plot_history(self):
        for i, dct in enumerate(self.__history):
            plt.subplot(2, 1, i + 1)
            for k, v in dct.items():
                plt.plot(v, label=k)
            if i == 1:
                plt.ylim((5, 9))


if __name__ == '__main__':
    N_ROOMS = 7
    print(f'{N_ROOMS=}')
    N_TIMESPANS = 9
    print(f'{N_TIMESPANS=}')

    s_timespan = np.arange(N_TIMESPANS)
    s_room = np.arange(N_ROOMS)
    # s_assign = s_timespan cross s_room

    df = pd.read_csv('test.csv', encoding='utf-8')

    m_band_to_name = {idx: row.iloc[0] for idx, row in df.iterrows()}
    s_band = np.arange(len(m_band_to_name))
    print(f'{len(s_band)=}')

    MAX_N_BAND_OCC = int((N_TIMESPANS * N_ROOMS) / len(s_band))
    print(f'{MAX_N_BAND_OCC=}')
    MAX_N_BANDS = MAX_N_BAND_OCC * len(s_band)
    print(f'{MAX_N_BANDS=}')

    m_person_to_name = {
        idx: name
        for idx, name in enumerate(sorted({
            x for x in df.iloc[:, 1:].to_numpy().flatten()
            if not pd.isnull(x)
        }))
    }
    s_person = np.arange(len(m_person_to_name))
    print(f'{len(s_person)=}')

    m_band_to_member = {
        band_name: np.array([m_person_to_name[idx] in member_names for idx in s_person]).astype(
            np.uint8)
        for band_name, member_names in zip(df.iloc[:, 0], df.iloc[:, 1:].to_numpy())
    }
    m_band_to_member = np.array([
        m_band_to_member[m_band_to_name[idx]]
        for idx in s_band
    ])


    def list_band_member_nodup(ss_band_rest=None, member_count=None, b_min=0):
        if ss_band_rest is None:
            ss_band_rest = np.ones_like(s_band, dtype=np.uint8)
        if member_count is None:
            member_count = np.zeros_like(s_person, dtype=np.int8)

        n = np.count_nonzero(ss_band_rest == 0)
        if n <= N_ROOMS:
            if N_ROOMS - 5 <= n <= N_ROOMS:
                yield np.uint8(ss_band_rest == 0).copy()
            b_set = np.where(ss_band_rest == 1)[0]
            for b in b_set:
                if b_min <= b:
                    new_member_count = member_count + m_band_to_member[b]
                    if np.all(new_member_count <= 1):
                        new_ss_band_rest = ss_band_rest.copy()
                        new_ss_band_rest[b] = 0
                        yield from list_band_member_nodup(new_ss_band_rest, new_member_count,
                                                          b_min=b + 1)


    print('listing search space up...')
    s_band_nodup_member = np.stack([v for v in list_band_member_nodup()])

    print('computing distance matrix...')
    dist_mat = cdist(s_band_nodup_member, s_band_nodup_member, metric='hamming')
    dist_mat *= s_band_nodup_member.shape[1]
    print(dist_mat.shape)

    hamming_1_or_2 = (dist_mat == 1) | (dist_mat == 2)

    del dist_mat


    @functools.lru_cache(maxsize=65536)
    def get_neighbours(b):
        a = np.where(hamming_1_or_2[b])[0]
        a.setflags(write=False)
        return a


    b_nodup_index = np.arange(len(s_band_nodup_member))


    def table_band_occurrence_count(table):
        occ_array = s_band_nodup_member[table].astype(np.int16).sum(axis=0)
        return occ_array


    def table_band_occurrence_score(table):
        occ_array = table_band_occurrence_count(table)
        return -np.mean(np.abs(occ_array - MAX_N_BAND_OCC))  # max is 0


def main():
    state = State.random_init()
    best = None
    bar = tqdm(range(64))
    for i_iter in bar:
        state.local_search()
        state.update_score_stat()
        state_best = state.best()
        if best is None or best.score < state_best.score:
            best = state_best
        state.reset_agents()

        # print('\n ----- ', i_iter)
        # print(state.scores().round(2))
        ci = get_neighbours.cache_info()
        bar.set_description(
            f'{best.description()} {state_best.description()} {ci.hits=} {ci.misses=} {ci.currsize=}')

        if best.root_occupation_rate() > 0.999:
            break

    plt.figure()
    state.plot_history()
    plt.legend()
    plt.show()

    for row in best.table():
        print(sorted(row))


if __name__ == '__main__':
    main()
