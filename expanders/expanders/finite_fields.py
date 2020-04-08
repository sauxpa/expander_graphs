import itertools
from functools import lru_cache
import numpy as np
import sympy
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_irreducible_p, gf_add, gf_sub, gf_mul,\
    gf_rem, gf_gcdex


class FiniteField():
    """Sympy's implementation of finite field is not exactly what it seems...
    For fields with higher extension degree (i.e of cardinal p^n with n>1),
    Sympy's GF(p^n) corresponds to Z/(p^n)Z which has not a field structure.
    Below implements an elementary finite field class with finite field
    arithmetic.

    The polynomials are represented by lists of coefficients, highest degrees
    first. For example, the elements of F_{3^2} are:
    [], [1], [2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2],
    with [] corresponding to the zero element, [x] to the elements of Z/pZ,
    the prime subfield of F_q.

    All credits to the very helpful thread on stackoverflow:
    https://stackoverflow.com/questions/48065360/
    interpolate-polynomial-over-a-finite-field/48067397#48067397.
    """
    def __init__(self, p: int, n: int = 1) -> None:
        p, n = int(p), int(n)

        self.check_p(p)
        self.check_n(n)

        self.flush()

        self._p = p
        self._n = n
        if n == 1:
            self.reducing = [1, 0]
        else:
            for c in itertools.product(range(p), repeat=n):
                poly = (1, *c)
                if gf_irreducible_p(poly, p, ZZ):
                    self.reducing = poly
                    break

    def check_p(self, p: int) -> None:
        assert sympy.isprime(p), 'p must be a prime number, not {}'.format(p)

    def check_n(self, n: int) -> None:
        assert n > 0, 'n must be a positive integer, not {}'.format(n)

    @property
    def p(self) -> int:
        return self._p

    @p.setter
    def p(self, new_p: int) -> None:
        self.check_p(new_p)
        self.flush()
        self._p = new_p

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, new_n: int) -> None:
        self.check_n(new_n)
        self.flush()
        self._n = new_n

    @property
    def q(self) -> int:
        return self.p ** self.n

    def flush(self) -> None:
        """Centralized call to flush all lru caches.
        """
        type(self).field.fget.cache_clear()

    @property
    @lru_cache(maxsize=None)
    def field(self) -> list:
        """List of elements in the finite field,
        presented as lists of coefficients of polynomials
        over the prime subfield F_p.
        """
        return [
            np.trim_zeros(list(c), trim='f') for c in itertools.product(
                range(self.p),
                repeat=self.n
                )
            ]

    @property
    def neutral(self) -> list:
        """Neutral element of the multiplicative group of the field invertibles.
        """
        return [1]

    def add(self, x: list, y: list) -> list:
        return gf_add(x, y, self.p, ZZ)

    def sub(self, x: list, y: list) -> list:
        return gf_sub(x, y, self.p, ZZ)

    def mul(self, x: list, y: list) -> list:
        return gf_rem(gf_mul(x, y, self.p, ZZ), self.reducing, self.p, ZZ)

    def power(self, x: list, k: int) -> list:
        """Fast exponentiation.
        """
        if k < 0:
            raise ValueError('Exponent must be a nonnegative integer, \
                             not {:s}'.format(k))
        elif k == 0:
            return self.neutral
        else:
            x_pow = self.power(x, k//2)
            x_pow = self.mul(x_pow, x_pow)
            if k % 2 == 1:
                x_pow = self.mul(x_pow, x)
            return x_pow

    def inv(self, x: list) -> list:
        s, t, h = gf_gcdex(x, self.reducing, self.p, ZZ)
        return s

    def eval_poly(self, poly: list, point: list) -> list:
        val = []
        for c in poly:
            val = self.mul(val, point)
            val = self.add(val, c)
        return val

    def legendre_symbol(self, x: list) -> int:
        """Extension of the standard Legendre symbol to the finite field F_q.
        (x | F_q) = 1 if x is a quadratic residue in F_q^*, -1 if not, and 0
        if x = 0.
        Euler's criterion (or simply counting the number of quadratic residues
        and applying Lagrange theorem on subgroup cardinals) implies that
        (x | F_q) = x ^ ((q-1)/2).
        """
        if len(x) == 0:
            return 0
        else:
            symbol = self.power(x, (self.q-1)//2)
            if len(symbol) > 1:
                raise Exception("Euler's criterion failed ({}^{} is \
                                of degree > 1)".format(x, (self.q-1)//2))
            else:
                if int(symbol[0]) == 1:
                    return 1
                elif int(symbol[0]) == self.p-1:
                    return -1
                else:
                    raise Exception("Euler's criterion failed ({}^{} is \
                                    not -1, 0 or 1)".format(x, (self.q-1)//2))
