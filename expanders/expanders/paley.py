import networkx as nx
import numpy as np
import sympy
from .expanders import GraphBuilder
from .finite_fields import FiniteField

def check_q(q: int) -> None:
    """Assert whether q is prime power.
    """
    assert q % 4 == 1, '{} != 1 mod 4'.format(q)
    if not sympy.isprime(q):
        split_power = sympy.perfect_power(q)
        assert split_power, '{} is not a power'.format(q)
        p, _ = split_power
        assert sympy.isprime(p), '{} is not prime'.format(p)


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

    @property
    def p(self):
        if sympy.isprime(self.q):
            return self.q
        else:
            return sympy.perfect_power(self.q)[0]

    @property
    def n(self):
        if sympy.isprime(self.q):
            return 1
        else:
            return sympy.perfect_power(self.q)[1]

    def _build(self) -> None:
        """Build Paley graph and store it in self.G.
        Nodes are elements of Fq and edges are (a,b) such that
        a-b is a square in Fq*.
        """
        self._G = nx.Graph()

        if sympy.isprime(self.q):
            # if q is a prime, just count the squares mod q...
            square_list = [(x ** 2) % self.q for x in range(1, self.q-1)]
            square_list = [x2 for x2 in square_list if x2 != 0]
            square_list = set(square_list)

            self._G.add_nodes_from(range(self.q))
            for x in range(self.q):
                for y in square_list:
                   self._G.add_edge(x, (x + y) % self.q)
        else:
            #... otherwise, q is a prime power and we need to count squares in
            # the finite field F_q.
            ff = FiniteField(self.p, self.n)
            square_list = [(ff.power(x, 2)) for x in ff.field if len(x) > 0]
            # Cannot hash list so cannot do set(list of list), so use this trick
            # to uniquify,
            square_list = [list(x2) for x2 in set(tuple(x2) for x2 in square_list)]

            self._G.add_nodes_from(range(self.q))
            for x in range(self.q):
                for y in square_list:
                    self._G.add_edge(x, ff.field.index(ff.add(ff.field[x], y)))

        self._G = nx.Graph(self._G)
