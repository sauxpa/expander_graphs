import abc
import networkx as nx
import numpy as np
from functools import lru_cache
import scipy
from typing import Union
from sklearn.metrics import mutual_info_score

def sorted_adjacency_spectrum(G: Union[nx.Graph, nx.MultiDiGraph]) -> np.ndarray:
    """Calculate adjacency spectrum of G
    (sorted in decreasing order).
    """
    _spectrum = nx.adjacency_spectrum(G)
    idx = _spectrum.argsort()[::-1]
    return np.real(_spectrum[idx])


def normalized_spectrum(P: scipy.sparse.csr.csr_matrix) -> np.ndarray:
    """Calculate the transition matrix spectrum. For d-regular graph, this is
    equivalent to scaling the adjacency spectrum by 1/d.
    """
    _spectrum, _ = np.linalg.eig(P.A)
    idx = _spectrum.argsort()[::-1]
    return np.real(_spectrum[idx])


def alon_boppana(d: int, normalized: bool=False) -> float:
    if normalized:
        return 2*np.sqrt(d-1)/d
    else:
        return 2*np.sqrt(d-1)


def spectral_gap(spectrum: np.ndarray) -> float:
    return spectrum[0] - spectrum[1]


def transition_matrix(G: Union[nx.Graph, nx.MultiDiGraph]) -> scipy.sparse.csr.csr_matrix:
    """Compute the transition matrix of the Markov standard
    random walk on graph G, i.e D^-1 * A where
    A is the adjacency matrix, D the diagonal matrix of degrees.
    """
    A = nx.adjacency_matrix(G)
    D = scipy.sparse.diags(1/A.dot(np.ones(A.shape[0])))
    return D.dot(A)


def invariant_distribution(P: scipy.sparse.csr.csr_matrix) -> np.ndarray:
    """Compute an invariant measure of transition matrix P i.e a left eigenvector
    associated with eigenvalue 1 (or equivalently an eigenvector of P.T for the same
    eigenvalue, the existence of which is guaranted by Perron-Frobenius theorem.)
    Returns the invariant probability distribution, i.e the normalized eigenvector.
    """
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(P.T, k=3)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:, idx])
    mu = eigenvectors[:, 0]
    mu /= np.sum(mu)
    return mu


def is_ramanujan(spectrum: np.ndarray) -> bool:
    """A connected d-regular graph is said to be Ramanujan when
    max_{i>=2} |lambda_i| <= 2*sqrt(d-1),
    where d = lambda_1 >= lambda_2 >= ... >= lambda_n are the adjacency eigenvalues.
    """
    return np.max(np.abs(spectrum[1:])) <= alon_boppana(spectrum[0])


def sample_random_walk(
    P: Union[np.ndarray, scipy.sparse.csr.csr_matrix],
    node: int,
    walk_length: int,
) -> np.ndarray:
    if scipy.sparse.issparse(P):
        P = P.A
    path = np.empty(walk_length)
    path[0] = node
    n = P.shape[0]
    for t in range(1, walk_length):
        node = int(np.random.choice(range(n), p=P[node, :].flatten()))
        path[t] = node
    return path


def mi_mixing(
    P : Union[np.ndarray, scipy.sparse.csr.csr_matrix],
    n_samples: int,
    walk_length: int,
) -> np.ndarray:
    if scipy.sparse.issparse(P):
        P = P.A
    n = P.shape[0]
    paths = np.empty((n_samples, walk_length))
    for i in range(n_samples):
        node = np.random.choice(range(n))
        path = sample_random_walk(P, node, walk_length)
        paths[i] = path

    mi = np.empty(walk_length)
    for t in range(walk_length):
        # mi[t] = normalized_mutual_info_score(paths[:, 0], paths[:, t], average_method='arithmetic')
        mi[t] = mutual_info_score(paths[:, 0], paths[:, t])
    return mi


class GraphBuilder(abc.ABC):
    def __init__(
        self,
    ) -> None:
        self._G = nx.Graph()

    def flush(self) -> None:
        """Centralized call to flush all lru caches.
        """
        type(self).spectrum.fget.cache_clear()
        type(self).normalized_spectrum.fget.cache_clear()
        type(self).transition_matrix.fget.cache_clear()
        type(self).invariant_distribution.fget.cache_clear()

    @property
    @lru_cache(maxsize=None)
    def spectrum(self) -> np.ndarray:
        """Calculate and cache adjacency spectrum
        (sorted in decreasing order).
        """
        return sorted_adjacency_spectrum(self.G)

    @property
    @lru_cache(maxsize=None)
    def normalized_spectrum(self) -> np.ndarray:
        """Calculate and cache transition spectrum
        (sorted in decreasing order).
        """
        return normalized_spectrum(self.transition_matrix)

    @property
    def is_ramanujan(self) -> bool:
        """A connected d-regular graph is said to be Ramanujan when
        max_{i>=2} |lambda_i| <= 2*sqrt(d-1),
        where d = lambda_1 >= lambda_2 >= ... >= lambda_n are the adjacency eigenvalues.
        """
        return is_ramanujan(self.spectrum)

    @property
    def spectral_gap(self) -> float:
        return spectral_gap(self.spectrum)

    @property
    def normalized_spectral_gap(self) -> float:
        return spectral_gap(self.normalized_spectrum)

    def sample_random_walk(self, node: int, walk_length: int) -> np.ndarray:
        return sample_random_walk(self.transition_matrix, node, walk_length)

    def mi_mixing(self, n_samples: int, walk_length: int) -> np.ndarray:
        return mi_mixing(self.transition_matrix, n_samples, walk_length)

    @property
    def G(self) -> Union[nx.Graph, nx.MultiDiGraph]:
        """Calculate and cache graph.
        """
        return self._G

    @property
    @lru_cache(maxsize=None)
    def transition_matrix(self) -> scipy.sparse.csr.csr_matrix:
        """Calculate and cache transition matrix.
        """
        return transition_matrix(self.G)

    @property
    @lru_cache(maxsize=None)
    def invariant_distribution(self) -> np.ndarray:
        """Calculate and cache invariant distribution.
        """
        return invariant_distribution(self.transition_matrix)

    def build(self) -> None:
        """Wraps _build method with centralized cache clearing.
        """
        self.flush()
        self._build()

    def _build(self) -> None:
        """Graph construction method to be defined in children classes.
        """
        pass
