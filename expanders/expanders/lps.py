import networkx as nx
import numpy as np
import sympy
from sympy.solvers.diophantine import power_representation
from sympy.utilities.iterables import signed_permutations
from .expanders import GraphBuilder


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
        self.check_p(p)
        self._p = p
        self._remove_parallel_edges = remove_parallel_edges
        self._remove_self_edges = remove_self_edges

        super().__init__()

    def check_p(self, p: int) -> None:
        """Assert whether p is prime.
        """
        assert sympy.isprime(p), '{} is not prime'.format(p)

    @property
    def p(self) -> int:
        return self._p
    @p.setter
    def p(self, new_p: int) -> None:
        self.check_p(new_p)
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


class LPS(GraphBuilder):
    """True Lubotsky-Phillips-Sarnak Cayley graph construction.
    """
    def __init__(
        self,
        p: int,
        q: int,
        remove_parallel_edges: bool=False,
        remove_self_edges: bool=False,
    ) -> None:
        self.check_p(p)
        self._p = p

        self.check_q(q)
        self._q = q

        self._remove_parallel_edges = remove_parallel_edges
        self._remove_self_edges = remove_self_edges


        super().__init__()

    def check_p(self, p: int) -> None:
        """Assert whether p is prime and with residue 1 mod 4.
        """
        assert sympy.isprime(p), '{} is not prime'.format(p)
        assert p % 4 == 1, '{} != 1 mod 4'.format(p)

    def check_q(self, q: int, p: int=None) -> None:
        """Assert whether q is a square mod p and has residue 1 mod 4.
        """
        if not p:
            p = self.p
        assert q % 4 == 1, '{} != 1 mod 4'.format(q)
        assert sympy.is_quad_residue(q, p), '{} must be a square mod {}'.format(q, p)

    @property
    def p(self) -> int:
        return self._p
    @p.setter
    def p(self, new_p: int) -> None:
        self.check_p(new_p)
        self.flush()
        self._p = new_p

    @property
    def q(self) -> int:
        return self._q
    @q.setter
    def q(self, new_q: int) -> None:
        self.check_q(new_q)
        self.flush()
        self._q = new_q

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

    def eligible_solution(self, sol: tuple) -> bool:
        """Filter solutions of a^2 + b^2 + c^2 + d^2 = p
        such that a is positive and odd.
        """
        return sol[0] % 2 == 1 and sol[0] > 0

    def find_square_root(self, a: int) -> int:
        """Find a number i in Fq such that i^2 = a.
        """
        for x in range(self.q):
            if x ** 2 % self.q == a % self.q:
                break
        else:
            raise Exception('-1 is not a square mod {}'.format(self.q))
        return x

    @property
    def infinity_point(self):
        return self.q

    def modular_inv(self, x):
        """Compute y such that x*y = 1 mod q
        (assuming x != 0).
        """
        return x ** (self.q-1) % self.q

    def _build(self) -> None:
        """Build Cayley graph and store it in self.G.
        """
        self._G = nx.MultiDiGraph()

        # Compute solutions to the four squares decomposition problem and retain
        # only the ones eligible for the LPS construction.
        four_squares = set([sol for x in power_representation(self.p, 2, 4, zeros=True) for sol in signed_permutations(x) if self.eligible_solution(sol)])

        i = self.find_square_root(-1)

        # One extra vertex for the infinity point
        for x in range(self.q+1):
            self._G.add_node(x)

        for x in range(self.q):
            for (a, b, c, d) in four_squares:
                num = ((a + i*b) * x + c + i*d) % self.q
                den = ((i*d - c) * x + a - i*b) % self.q
                if den == 0:
                    self._G.add_edge(x, self.infinity_point)
                else:
                    self._G.add_edge(x, num * self.modular_inv(den))

        # Finally add the links from the infinity point
        for (a, b, c, d) in four_squares:
            num = (a + i*b) % self.q
            den = (i*d - c) % self.q
            if den == 0:
                self._G.add_edge(self.infinity_point, self.infinity_point)
            else:
                self._G.add_edge(self.infinity_point, num * self.modular_inv(den))

        if self.remove_parallel_edges:
            self._G = nx.Graph(self._G)
            if self.remove_self_edges:
                self._G.remove_edges_from(self._G.selfloop_edges())
        else:
            self._G = nx.MultiDiGraph(self._G)
