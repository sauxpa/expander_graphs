import networkx as nx
import numpy as np
import sympy
from .expanders import GraphBuilder

def check_q(q: int) -> None:
    """Assert whether q is prime power.
    In theory Paley graphs can be defined on finite
    fields with q=p^k elements, but the construction
    requires to enumerate quadratic residues in the
    corresponding finite fields.
    For now, let's stick to fields of degree 1
    i.e Z/qZ with q prime.
    """
    assert q % 4 == 1, '{} != 1 mod 4'.format(q)
    if not sympy.isprime(q):
        raise Exception('Only prime numbers are allowed for now.')
        # split_power = sympy.perfect_power(q)
        # assert split_power, '{} is not a power'.format(q)
        # p, _ = split_power
        # assert sympy.isprime(p), '{} is not prime'.format(p)


class Paley(GraphBuilder):
    """Paley strongly regular dense graph.
    Pick number q=p^n where p is prime and q = 1 mod 4,
    such that -1 is a square in the finite field Fq.
    Build the graph (V, E) as follows:
    * V = Fq,
    * E = {(a,b) such that a-b is a square in Fq*}
    """
    def __init__(
        self,
        q: int,
    ) -> None:
        check_q(q)
        self._q = q

        super().__init__()

    @property
    def q(self) -> int:
        return self._q
    @q.setter
    def q(self, new_q: int) -> None:
        check_q(new_q)
        self.flush()
        self._q = new_q

    def _build(self) -> None:
        """Build Paley graph and store it in self.G.
        Nodes are elements of Fq and edges are (a,b) such that
        a-b is a square in Fq*.
        """
        self._G = nx.Graph()

        square_list = [(x ** 2) % self.q for x in range(1, (self.q-1) // 2)]
        square_list = [x2 for x2 in square_list if x2 != 0]
        square_list = set(square_list)

        self._G.add_nodes_from(range(self.q))
        for x in range(self.q):
            for y in square_list:
               self._G.add_edge(x, (x + y) % self.q)

        self._G = nx.Graph(self._G)
