import networkx as nx
from .expanders import GraphBuilder


class RRG(GraphBuilder):
    """Random Regular Graphself.
    Friedman showed in 2003 (https://arxiv.org/pdf/cs/0405020.pdf) that
    random regular graphs drawn from the uniform distribution are almost
    Ramanujan, in the sense that :
    for all eps>0, max_{i>=2} |lambda_i| <= 2*sqrt(d-1) + eps
    with probability 1-o(1) when n -> infty.
    """
    def __init__(
        self,
        d: int,
        n: int,
    ) -> None:
        # Degree
        self._n = n
        # Number of vertices
        self._d = d

        super().__init__()

    @property
    def d(self) -> int:
        return self._d

    @d.setter
    def d(self, new_d: int) -> None:
        self.flush()
        self._d = new_d

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, new_n: int) -> None:
        self.flush()
        self._n = new_n

    def _build(self) -> None:
        """
        """
        self._G = nx.random_regular_graph(self.d, self.n)
