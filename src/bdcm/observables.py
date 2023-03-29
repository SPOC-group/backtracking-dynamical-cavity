import numpy as np
from itertools import product
from scipy.optimize import fmin

from src.rules import DEAD, ALIFE

"""
Generic generator for observables from the paper.

You may define your own. All are functions of the form f(x_i,x_j) -> float, i.e. are edge localized. If we want to 
compute a node-localized functions, we therefore need to normalize correctly: We need to compute the value for both x_i 
and x_j. No averaging is done here, as this is done in the simulation code.
"""

MAG_cycle = ('m_attr', lambda config: lambda a,b: sum_(a[config['p']:],b[config['p']:]) )
MAG_path0 = ('m_init', lambda config: lambda a,b: sum_(a[0:1],b[0:1]) )
RAT_cycle = ('rho', lambda config: lambda a,b: rattling_(a[config['p']:],b[config['p']:]) )
Energy_cycle = ('energy_attr', lambda config: lambda a,b: energy(a[config['p']:config['p']+1],b[config['p']:config['p']+1]) )
Energy_pathend = ('energy_pathend', lambda config: lambda a,b: energy(a[config['p']-1:config['p']],b[config['p']-1:config['p']]) )

def energy(b,c):
    return  2 * (int(b == c) * 2 - 1) # number of monochromatic edges - number of bichromatics edges, each edge counted 2x

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def sum_(b, c):
    return ((sum(b) / len(b) * 2) - 1) + ((sum(c) / len(c) * 2) - 1)

def rattling_(b, c):
    return int(not all_equal(b)) + int(not all_equal(c))

class Observables:

    cycle_obs = [Energy_cycle, MAG_cycle]
    default_obs = [MAG_path0, Energy_pathend]
    path_obs = []
    def __init__(self, **config):

        self.obs = [] + self.default_obs
        if config['p'] > 0:
            self.obs += self.path_obs
        if config['c'] != 0:
            self.obs += self.cycle_obs
        if config['c'] > 1:
            self.obs += [RAT_cycle]

        self.obs_map = {v[0]: i for i, v in enumerate(self.obs)}
        self.names = [o[0] for o in self.obs]
        self.temps = np.array([config.get(name+'_temp',0.0) for name in self.names])
        self.funcs = [o[1](config) for o in self.obs]
        self.targets = np.array([config.get(name+'_target',0.0) for name in self.names])

        self.attr_size_graph = config['attr_size_graph']
        self.d = config['d']

    def measure(self,b,c):
        return np.array([f(b,c) for f in self.funcs])
    
    def g(self,b,c):
        return np.exp(-self.temps / self.d * self.measure(b,c))

    def Z_ij_prime_(self, chi):
        Z = np.zeros_like(self.temps)
        for c in product([DEAD, ALIFE], repeat=2 * self.attr_size_graph):
            b = [0] * len(c)
            # period stays the same, but direction is different: i->j, j->i
            b[::2] = c[1::2]
            b[1::2] = c[::2]
            b = tuple(b)
            Z += (chi[c] * chi[b]) * (1.0 / self.g(c[1::2],c[::2])).prod() * self.measure(c[1::2],c[::2])
        return Z

    def Z_ij_prime_pop(self, chi1,chi2):
        Z = np.zeros_like(self.temps)
        for c in product([DEAD, ALIFE], repeat=2 * self.attr_size_graph):
            b = [0] * len(c)
            # period stays the same, but direction is different: i->j, j->i
            b[::2] = c[1::2]
            b[1::2] = c[::2]
            b = tuple(b)
            Z += (chi1[c] * chi2[b]) * (1.0 / self.g(c[1::2],c[::2])).prod() * self.measure(c[1::2],c[::2])
        return Z

    def Z_ij_(self, chi):
        Z = 0
        for c in product([DEAD, ALIFE], repeat=2 * self.attr_size_graph):
            b = [0] * len(c)
            # period stays the same, but direction is different: i->j, j->i
            b[::2] = c[1::2]
            b[1::2] = c[::2]
            b = tuple(b)
            Z += (chi[c] * chi[b]) * (1.0 / self.g(c[1::2],c[::2])).prod()
        return Z

    def calc_marginals(self,chi):
        Z_ij_prime = self.Z_ij_prime_(chi)
        Z_ij = self.Z_ij_(chi)

        observables = 0.5 * Z_ij_prime / Z_ij
        return {
            **{name: o for name, o in zip(self.names, observables)},
            **{name + '_Z_ij_prime': t for name, t in zip(self.names,observables)},
            'Z_ij': Z_ij,
            'Legendre': (observables * self.temps).sum() # Legendre transform
        }

    def update_best_temps(self,observable, observable_target, chi):
        # optimize for the different functions
        idx = self.obs_map[observable]

        def f(temp):
            self.temps[idx] = temp
            Z_ij_prime = self.Z_ij_prime_(chi)
            Z_ij = self.Z_ij_(chi)
            observables = 0.5 * Z_ij_prime / Z_ij
            return (observable_target - observables[idx]) ** 2

        optimal_temp = fmin(f, 0.0, disp=False, ftol=0.00001, xtol=0.00001)[0]
        self.temps[idx] = -1 * 2 * optimal_temp 
