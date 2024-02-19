import hashlib
import os
import pickle
from typing import Any

import numpy as np


class Project:
    _PROJECT_ROOT = './projects'

    def __init__(self, domain: str, id_source: dict[str, Any]):
        id_source = sorted(id_source.items(), key=lambda x: x[0])
        self.__project_hash = hashlib.md5(str(id_source).encode('utf-8')).hexdigest()[2:]
        self.__path = os.path.join(self._PROJECT_ROOT, domain, self.__project_hash)
        os.makedirs(self.__path, exist_ok=True)

    def __repr__(self):
        return f'Project<{self.__path}>'

    def _path_to(self, data_name, extension):
        return os.path.join(self.__path, f'{data_name}.{extension}')

    def dump_pickle(self, data_name, data):
        path = self._path_to(data_name, 'pickle')
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, data_name):
        path = self._path_to(data_name, 'pickle')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def dump_numpy(self, data_name, data):
        path = self._path_to(data_name, 'npy')
        np.save(path, data)

    def load_numpy(self, data_name):
        path = self._path_to(data_name, 'npy')
        return np.load(path)

    def dump_mmap(self, data_name, data):
        path = self._path_to(data_name, 'mmap')
        self.dump_pickle(
            f'_info_{data_name}',
            dict(
                shape=data.shape,
                dtype=data.dtype
            )
        )
        return np.memmap(
            filename=path,
            mode='w+'
        )

    def load_mmap(self, data_name):
        path = self._path_to(data_name, 'mmap')
        info = self.load(f'_{data_name}_shape')
        # noinspection PyTypeChecker
        return np.memmap(
            filename=path,
            mode='r',
            dtype=info['dtype'],
            shape=info['shape']
        )

    def load(self, data_name):
        def iter_name_and_ext():
            for name in os.listdir(self.__path):
                yield os.path.splitext(name)

        def take_one():
            for file_name, ext in iter_name_and_ext():
                if file_name == data_name:
                    return file_name, ext
            raise ValueError('data not found', data_name)

        name, extension = self._path_to(*take_one())
        if extension == '.pickle':
            return self.load_pickle(name)
        elif extension == '.npy':
            return self.load_numpy(name)
        elif extension == '.mmap':
            return self.load_mmap(name)
        else:
            raise ValueError('invalid file format', data_name, extension)
