import networkx as nx
import numpy as np
import scipy
import sympy


def check_p(p: int) -> None:
    """Assert whether p is prime.
    """
    assert sympy.isprime(p), '{} is not prime'.format(p)


class LPS3():
    """3-regular expander graph based on a special case of
    Lubotsky-Phillips-Sarnak Cayley graph construction.
    Parallel edges (multiple edges x->y) and self edges (x->x) are
    unavoidable in the algebraic construction of LPS.
    To keep them (and enforce a 3-regular graph), use nx.MultiDiGraphself.
    To get rid of them, use remove_parallel_edges and remove_self_edges.
    """
    def __init__(
        self,
        p: int,
        remove_parallel_edges: bool=False,
        remove_self_edges: bool=False,
        ) -> None:

        check_p(p)
        self._p = p
        self._remove_parallel_edges = remove_parallel_edges
        self._remove_self_edges = remove_self_edges

        # Initialize graph and spectrum
        self.G = nx.MultiDiGraph()
        self.spectrum = np.empty(0)

    @property
    def p(self) -> int:
        return self._p
    @p.setter
    def p(self, new_p: int) -> None:
        check_p(new_p)
        self._p = new_p

    @property
    def remove_parallel_edges(self) -> bool:
        return self._remove_parallel_edges
    @remove_parallel_edges.setter
    def remove_parallel_edges(self, new_remove_parallel_edges: int) -> None:
        self._remove_parallel_edges = new_remove_parallel_edges

    @property
    def remove_self_edges(self) -> bool:
        return self._remove_self_edges
    @remove_self_edges.setter
    def remove_self_edges(self, new_remove_self_edges: int) -> None:
        self._remove_self_edges = new_remove_self_edges

    def build(self) -> None:
        """Build Cayley graph and store it in self.G.
        Each x != 0 is linked to x-1, x+1 and x^-1 mod p,
        0 is linked to itself, 1 and p-1.
        """
        _G = nx.MultiDiGraph()
        for x in range(self.p):
            _G.add_node(x)
        for x in range(self.p):
            if x == 0:
                _G.add_edge(0, self.p-1)
                _G.add_edge(0, 1)
                _G.add_edge(0, 0)
            else:
                _G.add_edge(x, x-1)
                _G.add_edge(x, (x+1) % self.p)
                _G.add_edge(x, (x**(self.p-2)) % self.p)

        if self.remove_parallel_edges:
            self.G = nx.Graph(_G)
            if self.remove_self_edges:
                self.G.remove_edges_from(self.G.selfloop_edges())
        else:
            self.G = nx.MultiDiGraph(_G)

        # Calculate and cache adjacency spectrum
        _spectrum = nx.adjacency_spectrum(self.G)
        idx = _spectrum.argsort()[::-1]
        self.spectrum = np.real(_spectrum[idx])

    @property
    def alon_boppana(self) -> float:
        return 2*np.sqrt(2)

    def assert_alon_boppana(self) -> float:
        assert np.max(np.abs(self.spectrum[1:])) <= self.alon_boppana, 'Alon-Boppana bound does not hold.'

    @property
    def spectral_gap(self):
        return self.spectrum[0] - self.spectrum[1]
