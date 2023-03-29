
import numpy as np

from math import sqrt
from numba import njit, vectorize, float64, uint8, typed, typeof

from pathlib import Path
import pickle
from collections import defaultdict
from tqdm import tqdm
import copy

from src.config import RESULT_DIR
from src.empirics.graph_samplers import random_regular_graph, random_regular_graph_config_model


# creates the connectivity graph
def get_neigh(d, n, seed,random_model='networkx'):
    neigh=np.zeros((n,d),dtype=np.int64)
    advancement=np.zeros(n).astype(int)
    if random_model == 'config':
        edges = random_regular_graph_config_model(d, n, seed=np.random.RandomState(seed))
    elif random_model == 'networkx':
        edges = random_regular_graph(d, n, seed=np.random.RandomState(seed))
    else:
        raise ValueError(f"Option {random_model} not supported.")
    for edge in edges:
        i, j = edge
        k = advancement[i]
        neigh[i,k] = j
        advancement[i] += 1
        k = advancement[j]
        neigh[j,k] = i
        advancement[j] += 1
    return neigh


# translates the string description of a rule to a numpy array, returns a jitted dictionary of the rule
def get_rule(rule):
    l=list(map(lambda x: x.replace("+", "2").replace("-", "3"), rule))
    l=np.asarray(l, dtype=np.uint8)
    return create_f_dict(l)


# create a dictionary with keys=(num of active nodes, state of current node), vals=0 or 1
@njit()
def create_f_dict(rule):
    f_dict=typed.Dict()
    i=uint8(0)
    for s in rule:
        if s in [0, 1]:
            f_dict[(uint8(i), uint8(0))]=uint8(s)
            f_dict[(uint8(i), uint8(1))]=uint8(s)
        elif s==2: #2 is a symbol for +
            f_dict[(uint8(i), uint8(0))]=uint8(0)
            f_dict[(uint8(i), uint8(1))]=uint8(1)
        elif s==3: #3 is a symbol for -
            f_dict[(uint8(i), uint8(0))]=uint8(1)
            f_dict[(uint8(i), uint8(1))]=uint8(0)
        i=i+1
    return f_dict

# iterate one time-step of the automaton
@njit()
def sim(f, n, neigh, from_config, to_config):
    for i in range(n):
        si=[from_config[j] for j in neigh[i]]
        s=uint8(sum(si))
        to_config[i]=f[(s, from_config[i])]

# compute the energy of the configuration
@njit()
def energy(n, neigh, signs, d):
    monchrom_edges = 0.0
    for i in range(n):
        monchrom_edges += sum([int(signs[i] == signs[j]) for j in neigh[i]])
    energy = - (1 - monchrom_edges/d/n) + monchrom_edges/d/n
    return energy

# compute the density of the configuration
@njit()
def get_density(n, signs):
    e = 0
    for i in range(n):
        e+=int(signs[i])
    return float(e)/n

# compute a transient of the dynamics
def get_trans(f, d, n, neigh, config,record_energy=False,record_density=False,t_max=np.inf):
    # t_max is the number of time-steps to simulate at most

    if t_max < 0:
        raise ValueError("t_max must be large enough")

    if t_max != np.inf:
        t_max = t_max + 2

    current_config = np.empty(n, dtype=np.uint8)
    history_m1_config = np.empty(n, dtype=np.uint8)
    history_m2_config = copy.deepcopy(config)

    sim(f, n, neigh, from_config=history_m2_config,to_config=history_m1_config)
    sim(f, n, neigh, from_config=history_m1_config, to_config=current_config)

    if record_energy:
        e = [energy(n, neigh, history_m2_config, d),energy(n, neigh, history_m1_config, d),energy(n, neigh, current_config, d)]
    else:
        e = None
    if record_density:
        dens = [history_m2_config.mean(),history_m1_config.mean(),current_config.mean()]
    else:
        dens = None


    t = 2
    if np.all(current_config==history_m1_config) and not np.all(current_config==history_m2_config):
        if t_max == 0:
            return None
        return [current_config], config, t-1, e, dens
    elif np.all(current_config==history_m2_config) and np.all(current_config==history_m1_config):
        return [current_config], config, t-2, e[:-1] if e is not None else e, dens[:-1] if dens is not None else dens
    elif np.all(current_config==history_m2_config) and not np.all(current_config==history_m1_config):
        return [history_m2_config,history_m1_config], config, t-2, e,dens
    if t_max == 0:
        return None
    while not (np.all(current_config==history_m1_config) or np.all(current_config==history_m2_config)):
        if t>t_max:
            # WE DO NOT HAVE AN ATTRACTOR YET, BUT REACHED THE MAXIMAL SIZE OF THE TRANSIENT (INCLUDING THE ATTRACTOR)
            return None
        t+=1

        # do one update step
        temp = history_m2_config
        history_m2_config = history_m1_config
        history_m1_config = current_config
        current_config = temp
        sim(f, n, neigh, from_config=history_m1_config, to_config=current_config)
        if record_energy:
            e.append(energy(n, neigh, current_config, d))
        if record_density:
            dens.append(current_config.mean())

    # we converged to an attractor
    if np.all(current_config==history_m1_config):
        if t == t_max:
            return None # for samplng correctly with a maximal transient, we cannot allow this as this would advantage 1-cycles.
        return [current_config],  config, t-1, e, dens
    return [history_m2_config,history_m1_config], config, t-2, e,dens



def simulate_varying_init(rule, n,path=None,samples=32,buckets=80,n_start=0,n_end=None,record_energy=False,random_model='networkx',record_density=False,t_max=np.inf):
    f = get_rule(rule)
    d = len(rule) - 1
    if path is None:
        path = f"results/d{d}_fb"
    if n_end is None:
        n_end = n

    densities=[int(np.round(i)) for i in np.linspace(n_start, n_end, num=buckets)]
    path = Path(path)
    path.mkdir(parents=True,exist_ok=True)
    rnd = np.random.randint(2 ** 32 - 1)
    path = path / f"{rule}_{n}_{rnd}.pkl"
    try:
        data=pickle.load(open(path, "rb"))
    except:
        data=[]

    # sample one graph per sample
    for i in tqdm(range(samples)):
        # compute only one graph
        seed = np.random.randint(2 ** 32 - 1)
        neigh = get_neigh(d, n, seed,random_model=random_model)

        for density in densities:
            config = np.random.permutation(
                np.concatenate((np.ones(density, dtype=np.uint8), np.zeros(n - density, dtype=np.uint8))))
            res = get_trans(f, d, n, neigh, config,record_density=record_density,record_energy=record_energy,t_max=t_max)
            while res is None or res[2] > t_max:
                config = np.random.permutation(
                np.concatenate((np.ones(density, dtype=np.uint8), np.zeros(n - density, dtype=np.uint8))))
                res = get_trans(f, d, n, neigh, config,record_density=record_density,record_energy=record_energy,t_max=t_max)
            attractor, init_config, transient, e, dens = res
            if len(attractor) == 2:
                static_nodes = len([j for j in range(n) if attractor[0][j] == attractor[1][j]])
            else:
                static_nodes=n
            data.append({"density": density,"density_history":dens,"energy_history":e,  "n": n, "trans": transient, 'static_nodes':static_nodes, 'attractor': len(attractor), 'densities': [get_density(n,a) for a in attractor], 'energies': [energy(n, neigh, a, d) for a in attractor],
                         "rattlers": 0.0 if len(attractor) == 1 else (attractor[0] != attractor[1]).sum()})
        if i % 20 == 0:
            pickle.dump(data, open(path, "wb"))
    pickle.dump(data, open(path, "wb"))

def get_data_small_graph(rule, n,path=None,samples=32,buckets=80,n_start=0,n_end=None,random_model='config',record_density=False):
    f = get_rule(rule)
    d = len(rule) - 1
    if path is None:
        path = f"results/d{d}_fb"
    if n_end is None:
        n_end = n

    random_model='config'
    densities=[int(np.round(i)) for i in np.linspace(n_start, n_end, num=buckets)]
    path = Path(path)
    path.mkdir(parents=True,exist_ok=True)
    rnd = np.random.randint(2 ** 32 - 1)
    path = path / f"{rule}_{n}_{rnd}.pkl"
    try:
        data=pickle.load(open(path, "rb"))
    except:
        data=[]

    # sample one graph per sample

    for i in tqdm(range(samples)):
        # compute only one graph
        seed = np.random.randint(2 ** 32 - 1)
        neigh = get_neigh(d, n, seed,random_model=random_model)

        for density in tqdm(densities):
            config = np.random.permutation(
                np.concatenate((np.ones(density, dtype=np.uint8), np.zeros(n - density, dtype=np.uint8))))
            attractor, init_config, transient, _, dens = get_trans(f, d, n, neigh, config,record_density=record_density)
            if len(attractor) == 2:
                static_nodes = len([j for j in range(n) if attractor[0][j] == attractor[1][j]])
            else:
                static_nodes=n
            data.append({"density": density,"density_history":dens, "n": n, "trans": transient, 'static_nodes':static_nodes, 'attractor': len(attractor), 'densities': [get_density(n,a) for a in attractor], 'energies': [energy(n, neigh, a, d) for a in attractor],
                         "rattlers": 0.0 if len(attractor) == 1 else (attractor[0] == attractor[1]).sum()})
        if i % 20 == 0:
            pickle.dump(data, open(path, "wb"))
    pickle.dump(data, open(path, "wb"))


def simulate_balanced_init(rule, n,path=None,samples=32,t_max=np.inf,density=None,random_model='networkx',record_energy=False,record_density=False):
    f = get_rule(rule)
    d = len(rule) - 1
    if density == None:
        density = int(n//2)

    if path is None:
        path = f"results/d{d}_fb"

    path = Path(path)
    path.mkdir(parents=True,exist_ok=True)
    rnd = np.random.randint(2 ** 32 - 1)
    path = path / f"{rule}_{n}_{rnd}.pkl"
    try:
        data=pickle.load(open(path, "rb"))
    except:
        data=[]

    # sample one graph per sample

    for i in tqdm(range(samples)):
        # compute only one graph
        seed = np.random.randint(2 ** 32 - 1)
        neigh = get_neigh(d, n, seed,random_model=random_model)

        config = np.random.permutation(
            np.concatenate((np.ones(density, dtype=np.uint8), np.zeros(n - density, dtype=np.uint8))))
        res = get_trans(f, d, n, neigh, config,record_density=record_density,record_energy=record_energy,t_max=t_max)
        while res is None or res[2] > t_max:
            config = np.random.permutation(
            np.concatenate((np.ones(density, dtype=np.uint8), np.zeros(n - density, dtype=np.uint8))))
            res = get_trans(f, d, n, neigh, config,record_density=record_density,record_energy=record_energy,t_max=t_max)
        attractor, init_config, transient, e, dens = res
        if len(attractor) == 2:
            static_nodes = len([j for j in range(n) if attractor[0][j] == attractor[1][j]])
        else:
            static_nodes=n
        data.append({"density": density,"density_history":dens,"energy_history":e,  "n": n, "trans": transient, 'static_nodes':static_nodes, 'attractor': len(attractor), 'densities': [get_density(n,a) for a in attractor], 'energies': [energy(n, neigh, a, d) for a in attractor],
                    "rattlers": 0.0 if len(attractor) == 1 else (attractor[0] != attractor[1]).sum()})
        if i % 100 == 0:
            pickle.dump(data, open(path, "wb"))
    pickle.dump(data, open(path, "wb"))