import codecs
import functools
import json

import numpy as np
from scipy.spatial.distance import cdist

from project import Project


class ProjectData:
    def __init__(self, input_json):
        self.__cache = dict(input_json=input_json)

    @property
    def input_json(self):
        return self.__cache['input_json']

    @classmethod
    def from_input_json_path(cls, input_json_path):
        with codecs.open(input_json_path, 'r', encoding='utf-8') as f:
            input_json = json.load(f)

        return ProjectData(input_json)

    @classmethod
    def from_cache(cls, input_json_path):
        project_data_nocache = cls.from_input_json_path(input_json_path)
        project_data_nocache.load_cache()
        return project_data_nocache

    @property
    def project_id_source(self) -> dict:
        return self.input_json['config']

    @functools.cached_property
    def project(self):
        return Project(domain='studio_scheduling', id_source=self.project_id_source)

    @functools.cached_property
    def member_names(self):  # -> ndarray[str, (N_MEMBER,)]
        member_names = {
            member_name
            for band in self.input_json['band_list']
            for member_name in band['member_names']
        }
        a = np.array(sorted(member_names))
        a.setflags(write=False)
        return a

    @functools.cached_property
    def band_names(self):  # -> ndarray[str, (N_BAND,)]
        band_names = {
            band['band_name']
            for band in self.input_json['band_list']
        }
        a = np.array(sorted(band_names))
        a.setflags(write=False)
        return a

    @functools.cached_property
    def n_timespan(self) -> int:
        return self.input_json['config']['n_timespan']

    @functools.cached_property
    def n_room(self) -> int:
        return self.input_json['config']['n_room']

    @functools.cached_property
    def n_band(self) -> int:
        return len(self.band_names)

    @functools.cached_property
    def n_member(self) -> int:
        return len(self.member_names)

    @functools.cached_property
    def band_to_member_set(self):  # -> ndarray[bool, (N_BAND, N_MEMBER)]
        a = np.array([
            [
                member_name in band['member_names']
                for member_name in self.member_names
            ]
            for band in self.input_json['band_list']
        ])
        a.setflags(write=False)
        return a

    def _band_set_space(self):  # -> ndarray[bool, (N_BAND_SET, N_BAND)]
        print('creating band set space...')

        def list_band_set_nodup_member(ss_band_rest=None, member_count=None, b_min=0):
            if ss_band_rest is None:
                ss_band_rest = np.ones(self.n_band, dtype=np.uint8)
            if member_count is None:
                member_count = np.zeros(self.n_member, dtype=np.int8)

            n = np.count_nonzero(ss_band_rest == 0)
            if n <= self.n_room:
                if self.n_room - 5 <= n <= self.n_room:
                    yield np.uint8(ss_band_rest == 0).copy()
                b_set = np.where(ss_band_rest == 1)[0]
                for b in b_set:
                    if b_min <= b:
                        new_member_count = member_count + self.band_to_member_set[b]
                        if np.all(new_member_count <= 1):
                            new_ss_band_rest = ss_band_rest.copy()
                            new_ss_band_rest[b] = 0
                            yield from list_band_set_nodup_member(
                                new_ss_band_rest,
                                new_member_count,
                                b_min=b + 1
                            )

        return np.stack(list(list_band_set_nodup_member())).astype(bool)

    @functools.cached_property
    def band_set_space(self):  # band set arrays of no member duplication
        if 'band_set_space' not in self.__cache:
            self.__cache['band_set_space'] = self._band_set_space()
        a = self.__cache['band_set_space']
        a.setflags(write=False)
        return a

    @functools.cached_property
    def n_band_set(self) -> int:
        return len(self.band_set_space)

    def _band_set_neighbour_indexes(self):  # -> list with N_BAND_SET items of ndarray[int, (None,)]
        print('creating band set neighbour indexes...')
        dist_mat = cdist(self.band_set_space, self.band_set_space, metric='hamming')
        dist_mat *= self.n_band
        hamming_1_or_2 = (dist_mat == 1) | (dist_mat == 2)
        # noinspection PyUnresolvedReferences
        return [
            np.where(hamming_1_or_2[i])[0]
            for i in range(self.n_band_set)
        ]

    @functools.cached_property
    def band_set_neighbour_indexes(self):  # band set arrays of index to its neighbours
        if 'band_set_neighbour_indexes' not in self.__cache:
            self.__cache['band_set_neighbour_indexes'] = self._band_set_neighbour_indexes()
        a = self.__cache['band_set_neighbour_indexes']
        for item in a:
            item.setflags(write=False)
        return a

    @functools.lru_cache(maxsize=65536)
    def get_neighbours(self, band_set_index):
        return self.band_set_neighbour_indexes[band_set_index]

    def dump_cache(self):
        _ = self.band_set_space
        _ = self.band_set_neighbour_indexes
        self.project.dump_pickle('cache', self.__cache)

    def load_cache(self):
        self.__cache = self.project.load_pickle('cache')

    def table_band_occurrence_count(self, band_set_indexes):
        band_occ_count = self.band_set_space[band_set_indexes].sum(axis=0)
        return band_occ_count

    @functools.cached_property
    def n_max_band_occurrence(self):
        return int(self.n_timespan * self.n_room) // self.n_band

    @functools.cached_property
    def n_max_total_bands(self):
        return self.n_max_band_occurrence * self.n_band

    def band_occupation_rate(self, band_set_indexes):
        band_occ_count = self.table_band_occurrence_count(band_set_indexes)
        band_occ_count = np.clip(band_occ_count, 0, self.n_max_band_occurrence)
        n_total_band_occ = band_occ_count.sum()
        return n_total_band_occ / self.n_max_total_bands
