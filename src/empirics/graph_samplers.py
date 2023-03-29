from collections import defaultdict
import numpy as np

def random_regular_graph(d, n, seed=None):
    """
    Returns a random regular graph of degree `d` and `n` nodes.

    Implementation adapted from NetworkX.
    """

    if (n * d) % 2 != 0:
        raise ValueError("n * d must be even")

    if not 0 <= d < n:
        raise ValueError("the 0 <= d < n inequality must be satisfied")
    def _suitable(edges, potential_edges):
        # Helper subroutine to check if there are suitable edges remaining
        # If False, the generation of the graph has failed
        if not potential_edges:
            return True
        for s1 in potential_edges:
            for s2 in potential_edges:
                # Two iterators on the same dictionary are guaranteed
                # to visit it in the same order if there are no
                # intervening modifications.
                if s1 == s2:
                    # Only need to consider s1-s2 pair one time
                    break
                if s1 > s2:
                    s1, s2 = s2, s1
                if (s1, s2) not in edges:
                    return True
        return False

    def _try_creation():
        # Attempt to create an edge set

        edges = set()
        stubs = list(range(n)) * d

        while stubs:
            potential_edges = defaultdict(lambda: 0)
            seed.shuffle(stubs)
            stubiter = iter(stubs)
            for s1, s2 in zip(stubiter, stubiter):
                if s1 > s2:
                    s1, s2 = s2, s1
                if s1 != s2 and ((s1, s2) not in edges):
                    edges.add((s1, s2))
                else:
                    potential_edges[s1] += 1
                    potential_edges[s2] += 1

            if not _suitable(edges, potential_edges):
                return None  # failed to find suitable edge set

            stubs = [
                node
                for node, potential in potential_edges.items()
                for _ in range(potential)
            ]
        return edges

    # Even though a suitable edge set exists,
    # the generation of such a set is not guaranteed.
    # Try repeatedly to find one.
    edges = _try_creation()
    while edges is None:
        edges = _try_creation()

    return edges

def random_regular_graph_config_model(d, n, seed=None):

    """
    Configuration model for random regular graphs.
    """

    def try_creation():
        edges = set()
        stubs = list(range(n)) * d

        seed.shuffle(stubs)
        stubiter = iter(stubs)
        for s1, s2 in zip(stubiter, stubiter):
            if s1 > s2:
                s1, s2 = s2, s1

            if s1 != s2 and ((s1, s2) not in edges):
                edges.add((s1, s2))
            else:
                return None

        return edges

    edges = try_creation()
    while edges is None:
        edges = try_creation()

    return edges