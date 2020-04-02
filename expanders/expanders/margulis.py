import networkx as nx
import numpy as np
import itertools
from .expanders import GraphBuilder

class Margulis(GraphBuilder):
    """Margulis degree 8 construction.
    Take the discrete torus (Z/nZ) x (Z/nZ) as the set of vertices,
    and link every vertex (x, y) to:
    (x +/- 2 * y, y),
    (x +/- (2 * y + 1), y),
    (x, y +/- 2 * x),
    (x, y +/- (2 * x + 1)).
    This results in a 8-regular graph on n^2 vertices.
    """
    def __init__(
        self,
        n: int,
        remove_parallel_edges: bool=False,
        remove_self_edges: bool=False,
        ) -> None:
        self._n = n
        self._remove_parallel_edges = remove_parallel_edges
        self._remove_self_edges = remove_self_edges

        super().__init__()

    @property
    def n(self) -> int:
        return self._n
    @n.setter
    def n(self, new_n: int) -> None:
        check_n(new_n)
        self.flush()
        self._n = new_n

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

    def margulis_bound(self, normalized=False):
        """The second largest eigenvalue is lower than thisself.
        """
        if normalized:
            return 5 * np.sqrt(2) / 8
        else:
            return 5 * np.sqrt(2)

    def _build(self) -> None:
        """
        """
        self._G = nx.MultiDiGraph()

        # Add nodes first...
        for (x, y) in itertools.product(range(self.n), repeat=2):
            self._G.add_node((x, y))

        # ... then the edges
        for (x, y) in itertools.product(range(self.n), repeat=2):
            self._G.add_edge((x, y), ((x + 2 * y) % self.n, y))
            self._G.add_edge((x, y), ((x - 2 * y) % self.n, y))
            self._G.add_edge((x, y), ((x + 2 * y + 1) % self.n, y))
            self._G.add_edge((x, y), ((x - 2 * y - 1) % self.n, y))
            self._G.add_edge((x, y), (x, (y + 2 * x) % self.n))
            self._G.add_edge((x, y), (x, (y - 2 * x) % self.n))
            self._G.add_edge((x, y), (x, (y + 2 * x + 1) % self.n))
            self._G.add_edge((x, y), (x, (y - 2 * x - 1) % self.n))

        if self.remove_parallel_edges:
            self._G = nx.Graph(self._G)
            if self.remove_self_edges:
                self._G.remove_edges_from(self._G.selfloop_edges())
        else:
            self._G = nx.MultiDiGraph(self._G)
