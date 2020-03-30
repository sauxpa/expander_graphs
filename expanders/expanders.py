import abc
import networkx as nx
import numpy as np
from functools import lru_cache
from typing import Union


class GraphBuilder(abc.ABC):
    def __init__(
        self,
    ) -> None:
        self._G = nx.Graph()

    def flush(self) -> None:
        """Centralized call to flush all lru caches.
        """
        type(self).spectrum.fget.cache_clear()

    @property
    @lru_cache(maxsize=None)
    def spectrum(self) -> np.ndarray:
        """Calculate and cache adjacency spectrum
        (sorted in decreasing order).
        """
        _spectrum = nx.adjacency_spectrum(self.G)
        idx = _spectrum.argsort()[::-1]
        return np.real(_spectrum[idx])

    def alon_boppana(self, d) -> float:
        return 2*np.sqrt(d-1)

    @property
    def is_ramanujan(self) -> bool:
        """A connected d-regular graph is said to be Ramanujan when
        max_{i>=2} |lambda_i| <= 2*sqrt(d-1),
        where d = lambda_1 >= lambda_2 >= ... >= lambda_n are the adjacency eigenvalues.
        """
        return np.max(np.abs(self.spectrum[1:])) <= self.alon_boppana(self.spectrum[0])

    @property
    def spectral_gap(self) -> float:
        return self.spectrum[0] - self.spectrum[1]

    @property
    def G(self) -> Union[nx.Graph, nx.MultiDiGraph]:
        """Calculate and cache graph.
        """
        return self._G

    def build(self) -> None:
        """Wraps _build method with centralized cache clearing.
        """
        self.flush()
        self._build()

    def _build(self) -> None:
        """Graph construction method to be defined in children classes.
        """
        pass
