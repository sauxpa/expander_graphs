import networkx as nx
import numpy as np
import sympy
from .expanders import GraphBuilder

def check_p(p: int) -> None:
    """Assert whether p is prime.
    """
    assert sympy.isprime(p), '{} is not prime'.format(p)


class LPS3(GraphBuilder):
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

        super().__init__()

    @property
    def p(self) -> int:
        return self._p
    @p.setter
    def p(self, new_p: int) -> None:
        check_p(new_p)
        self.flush()
        self._p = new_p

    @property
    def remove_parallel_edges(self) -> bool:
        return self._remove_parallel_edges
    @remove_parallel_edges.setter
    def remove_parallel_edges(self, new_remove_parallel_edges: int) -> None:
        self.flush()
        self._remove_parallel_edges = new_remove_parallel_edges

    @property
    def remove_self_edges(self) -> bool:
        return self._remove_self_edges
    @remove_self_edges.setter
    def remove_self_edges(self, new_remove_self_edges: int) -> None:
        self.flush()
        self._remove_self_edges = new_remove_self_edges

    def _build(self) -> None:
        """Build Cayley graph and store it in self.G.
        Each x != 0 is linked to x-1, x+1 and x^-1 mod p,
        0 is linked to itself, 1 and p-1.
        """
        self._G = nx.MultiDiGraph()
        for x in range(self.p):
            self._G.add_node(x)
        for x in range(self.p):
            if x == 0:
                self._G.add_edge(0, self.p-1)
                self._G.add_edge(0, 1)
                self._G.add_edge(0, 0)
            else:
                self._G.add_edge(x, x-1)
                self._G.add_edge(x, (x+1) % self.p)
                self._G.add_edge(x, (x**(self.p-2)) % self.p)

        if self.remove_parallel_edges:
            self._G = nx.Graph(self._G)
            if self.remove_self_edges:
                self._G.remove_edges_from(self._G.selfloop_edges())
        else:
            self._G = nx.MultiDiGraph(self._G)
