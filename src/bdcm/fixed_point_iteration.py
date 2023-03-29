import numpy as np
from itertools import product
from tqdm import tqdm

from src.rules import TotalisiticRule, DEAD, ALIFE
from src.bdcm.initialization import init_random_Gauss
from src.bdcm.observables import Observables


def Z_i_(chi, d, allowed_alife_neighbours, attr_size_graph, attractor_graph,homogenous_point_attractor,require_full_rattling,all_rattlers):
    Z = 0
    shape = [0] * (attr_size_graph * 2)
    shape[:attr_size_graph] = [2] * attr_size_graph
    shape[attr_size_graph:] = [d + 1] * attr_size_graph
    ALL_ALLOWED = np.arange(d+1)
    prob_alive = np.zeros(shape)
    # what is the probability of having k_1 alife neighbours in period p_1 and k_2 in p_2 if x is dead? prob_alive[DEAD, k1,k2]
    for c in product([DEAD, ALIFE], repeat=attr_size_graph):
        for x in product([DEAD, ALIFE], repeat=attr_size_graph):
            idx = [0] * attr_size_graph * 2
            idx[OUT::2] = c
            idx[IN::2] = x
            idx = tuple(idx)

            p_idx = [0] * (attr_size_graph * 2)
            p_idx[:attr_size_graph] = x
            p_idx[attr_size_graph:] = c
            p_idx = tuple(p_idx)
            prob_alive[p_idx] += chi[idx]
    for k in range(1, d):
        prob_alive__ = np.zeros_like(prob_alive)
        for c in product([DEAD, ALIFE], repeat=attr_size_graph):
            for x in product([DEAD, ALIFE], repeat=attr_size_graph):
                idx = [0] * attr_size_graph * 2
                idx[OUT::2] = c
                idx[IN::2] = x
                idx = tuple(idx)

                p_idx = [0] * (attr_size_graph * 2)
                p_idx[:attr_size_graph] = x
                p_idx[attr_size_graph:] = [slice(c[p], k + 1 + c[p]) for p in range(attr_size_graph)]
                p_idx = tuple(p_idx)

                o_idx = [0] * (attr_size_graph * 2)
                o_idx[:attr_size_graph] = x
                o_idx[attr_size_graph:] = [slice(None, k + 1) for p in range(attr_size_graph)]
                o_idx = tuple(o_idx)

                prob_alive__[p_idx] += prob_alive[o_idx] * chi[idx]
        prob_alive = prob_alive__

    for c in product([DEAD, ALIFE], repeat=attr_size_graph):
        allowals = []

        if all_rattlers is not None:
            for curr, nex in enumerate(attractor_graph[:-1]):
                allowed, change_occured = check_Nx_allowance_Z(c, curr, nex, allowed_alife_neighbours, d)
                allowals.append(allowed)
            current_OUT = c[-1]
            next_OUT = 0 if c[-1] == 1 else 1

            allowed = allowed_alife_neighbours[current_OUT]

            # if change occurs, we need to change the neighbour
            change_occurs = current_OUT != next_OUT
            if change_occurs:
                # we actually want the node to change their color, so we need to make them unahppy in this step!
                allowed = not_(allowed, d + 1)

            allowals.append(allowed)
        elif homogenous_point_attractor is not None:
            for curr, nex in enumerate(attractor_graph[:-1]):
                allowed,change_occured = check_Nx_allowance_Z(c, curr, nex, allowed_alife_neighbours, d)
                allowals.append(allowed)
            current_OUT = c[-1]
            next_OUT = homogenous_point_attractor

            allowed = allowed_alife_neighbours[current_OUT]

            # if change occurs, we need to change the neighbour
            change_occurs = current_OUT != next_OUT
            if change_occurs:
                # we actually want the node to change their color, so we need to make them unahppy in this step!
                allowed = not_(allowed, d + 1)

            allowals.append(allowed)
        else:
            for curr, nex in enumerate(attractor_graph):
                if nex is None:
                    allowals.append(ALL_ALLOWED)
                    continue
                allowed,change_occured = check_Nx_allowance_Z(c, curr, nex, allowed_alife_neighbours, d)
                if curr == len(attractor_graph) - 1 and require_full_rattling and not change_occured:
                    allowals.append([])
                else:
                    allowals.append(allowed)

        Z += prob_alive[tuple(c)][np.ix_(*allowals)].sum()

    return Z

def check_Nx_allowance_Z(node, current_step, next_step, allowed_alife_neighbours, d):
    current_OUT = node[current_step]
    next_OUT = node[next_step]

    allowed = allowed_alife_neighbours[current_OUT]

    # if change occurs, we need to change the neighbour
    change_occurs = current_OUT != next_OUT
    if change_occurs:
        # we actually want the node to change their color, so we need to make them unahppy in this step!
        allowed = not_(allowed, d + 1)

    return allowed, change_occurs


def not_(allowed, d):
    return np.delete(np.arange(d), allowed)

from functools import lru_cache


def check_Nx_allowance(NODES, current_step, next_step, func):
    current = NODES[2 * current_step:2 * current_step + 2]
    next = NODES[2 * next_step:2 * next_step + 2]

    return func(current[OUT], current[IN], next[OUT])


OUT = 0
IN = 1


def run_bdcm(its, rule, d, c=1, p=0,
                    init_func=init_random_Gauss, alpha=0.99,
                    fix_observable=None, balance_colors=False,
                    all_rattlers=None,homogenous_point_attractor=None,
                    require_full_rattling=False,epsilon=None,**kwargs ):

    if homogenous_point_attractor is not None:
        assert homogenous_point_attractor in [0,1]
        assert c == 1 
        c = 0 # we do not need to count the OUT node

    if all_rattlers is not None:
        assert c == 2
        c = 1

    if epsilon is None:
        # stick to iterations and do not break on convergence
        epsilon = 10e-20

    if c==0:
        has_cycle = False
    else:
        has_cycle = True

    # preprocess configuration
    attr_size_graph = c + p
    if fix_observable is not None:
        target_observable, target_value = fix_observable

    config = {**locals(), **kwargs}
    del config['kwargs']

    ALL_ALLOWED = np.arange(d)

    config['init_func'] = config['init_func'].__name__

    rule = TotalisiticRule(rule,d)
    allowed_alife_neighbours = rule.allowed_alife_neighbours
    observables = Observables(**config)
    attractor_graph = [(p +1) % attr_size_graph for p in range(attr_size_graph)]
    attractor_graph[-1] = p if has_cycle else None # also allow to not have cycles

    shape = tuple([2] * (attr_size_graph * 2))
    chi = init_func(shape)
    chi /= chi.sum()


    init_chi = chi.copy()

    # create function helper that is chached
    @lru_cache(maxsize=None)
    def determine_allowals(current_out, current_in, next_out):
        # outgoing node -> incoming node
        allowed = allowed_alife_neighbours[current_out] - int(current_in == ALIFE)
        allowed = allowed[allowed <= d - 1]
        allowed = allowed[allowed >= 0]

        # if change occurs, we need to change the neighbour
        change_occurs = current_out != next_out
        if change_occurs:
            # we actually want the node to change their color, so we need to make them unahppy in this step!
            allowed = not_(allowed, d)

        return allowed, change_occurs

    # direction: x->y
    i = 0
    converged = False
    pbar = tqdm(total=its)
    while i < its and not converged:
        i+=1

        # todo extend to all observables
        if fix_observable is not None:

            observables.update_best_temps(target_observable,target_value,chi)


        shape = tuple([2] * (attr_size_graph * 2))
        chi_dp = np.zeros(shape)

        shape = [0] * (attr_size_graph * 2)
        shape[:attr_size_graph] = [2] * attr_size_graph # first part is the state of the center node
        shape[attr_size_graph:] = [d] * attr_size_graph # second part is the number of alive neighbours in the neighbourhood


        # the probability is defines as follows:
        # what is the probability of having k_1 alife neighbours in period p_1 and k_2 in p_2 if x is dead in both? prob_alive[DEAD,DEAD, k1,k2]
        prob_alive = np.zeros(shape)

        # STEP Initialize the DP probability calculation
        for OUT_NODE in product([DEAD, ALIFE], repeat=attr_size_graph):
            for IN_NODE in product([DEAD, ALIFE], repeat=attr_size_graph):
                idx = [0] * attr_size_graph * 2
                idx[OUT::2] = OUT_NODE
                idx[IN::2] = IN_NODE
                idx = tuple(idx)

                p_idx = [0] * (attr_size_graph * 2)
                p_idx[:attr_size_graph] = IN_NODE
                p_idx[attr_size_graph:] = OUT_NODE
                p_idx = tuple(p_idx)
                prob_alive[p_idx] += chi[idx]

        # STEP Do the DP, this does not incorporate any constraints on the cycles
        for k in range(1, d - 1):
            prob_alive__ = np.zeros_like(prob_alive)
            for OUT_NODE in product([DEAD, ALIFE], repeat=attr_size_graph):
                for IN_NODE in product([DEAD, ALIFE], repeat=attr_size_graph):

                    idx = [0] * attr_size_graph * 2
                    idx[OUT::2] = OUT_NODE
                    idx[IN::2] = IN_NODE
                    idx = tuple(idx)

                    p_idx = [0] * (attr_size_graph * 2)
                    p_idx[:attr_size_graph] = IN_NODE
                    p_idx[attr_size_graph:] = [slice(OUT_NODE[p], k + 1 + OUT_NODE[p]) for p in range(attr_size_graph)]
                    p_idx = tuple(p_idx)

                    o_idx = [0] * (attr_size_graph * 2)
                    o_idx[:attr_size_graph] = IN_NODE

                    o_idx[attr_size_graph:] = [slice(None, k + 1) for p in range(attr_size_graph)]
                    o_idx = tuple(o_idx)

                    prob_alive__[p_idx] += prob_alive[o_idx] * chi[idx]
            prob_alive = prob_alive__

        # Step to sum only over the allowed combinations for each chi, this changes depending on whether we care about cycles, or cycles with paths as attractors
        for NODES in product([DEAD, ALIFE], repeat=2 * attr_size_graph):
            allowals = []
            IN_NODE = NODES[IN::2]
            OUT_NODE = NODES[OUT::2]
            


            if homogenous_point_attractor is not None:
                for curr, nex in enumerate(attractor_graph[:-1]):
                    allowed, change_occured = check_Nx_allowance(NODES, curr, nex, determine_allowals)
                    allowals.append(allowed)
                current_step = len(attractor_graph) - 1
                # for the last state, check that it goes into [1,1]
                current = NODES[2 * current_step:2 * current_step + 2]
                next = [homogenous_point_attractor,homogenous_point_attractor]

                allowed = allowed_alife_neighbours[current[OUT]] - int(current[IN] == ALIFE)
                allowed = allowed[allowed <= d - 1]
                allowed = allowed[allowed >= 0]

                # if change occurs, we need to change the neighbour
                change_occurs = current[OUT] != next[OUT]
                if change_occurs:
                    # we actually want the node to change their color, so we need to make them unahppy in this step!
                    allowed = not_(allowed, d)
                allowals.append(allowed)
            elif all_rattlers is not None:
                for curr, nex in enumerate(attractor_graph[:-1]):
                    allowed, change_occured = check_Nx_allowance(NODES, curr, nex, determine_allowals)
                    allowals.append(allowed)
                current_step = len(attractor_graph) - 1

                opp = lambda x: 0 if x == 1 else 1
                # for the last state, check that it goes into the opposite
                current = NODES[2 * current_step:2 * current_step + 2]
                next = [opp(current[0]),opp(current[1])]

                allowed = allowed_alife_neighbours[current[OUT]] - int(current[IN] == ALIFE)
                allowed = allowed[allowed <= d - 1]
                allowed = allowed[allowed >= 0]

                # if change occurs, we need to change the neighbour
                change_occurs = current[OUT] != next[OUT]
                if change_occurs:
                    # we actually want the node to change their color, so we need to make them unahppy in this step!
                    allowed = not_(allowed, d)
                allowals.append(allowed)

            else:
                for curr, nex  in enumerate(attractor_graph):
                    if nex is None:
                        allowals.append(ALL_ALLOWED)
                        continue

                    allowed, change_occured = check_Nx_allowance(NODES, curr, nex,determine_allowals )
                    if curr == len(attractor_graph) - 1 and require_full_rattling and not change_occured:
                        allowals.append([]) # all are allowed where a change occured?
                    else:
                        allowals.append(allowed)

            # c[::2] will be the center node over all attr_size_graph
            chi_dp[tuple(NODES)] = prob_alive[tuple(OUT_NODE)][np.ix_(*allowals)].sum()


        # STEP to append the priors to the distribution
        for x_center in product([DEAD, ALIFE], repeat=attr_size_graph):
            for x_parent in product([DEAD, ALIFE], repeat=attr_size_graph):
                OUT_NODE = [0] * (attr_size_graph * 2)
                OUT_NODE[::2] = x_center
                OUT_NODE[1::2] = x_parent
                chi_dp[tuple(OUT_NODE)] *= observables.g(x_center,x_parent).prod()

        if balance_colors:
            for NODES in product([DEAD, ALIFE], repeat=2 * attr_size_graph):
                # set the inputs such that the inverse of every item is equal to its color symmetric item
                SYM_NODES = tuple(0 if kk == 1 else 1 for kk in NODES)
                temp = (chi_dp[tuple(NODES)] + chi_dp[SYM_NODES]) / 2
                chi_dp[tuple(NODES)] = temp
                chi_dp[tuple(SYM_NODES)] = temp


        chi__ = chi_dp.copy()
        chi__ = chi__ / chi__.sum()
        chi_old = chi.copy()
        chi = alpha * chi + (1 - alpha) * chi__
        dist = abs(chi_old - chi__).sum()
        if i > 10 and dist < epsilon:
            pbar.set_description(f'[Fixed Point It.] acc={dist:.16f} |')
            pbar.close()
            print(f'Stopping early because of convergence at iteration {i} with distance {dist:.16f}.')
            converged=True
            break

        if i % 100 == 1:
            pbar.set_description(f'[Fixed Point It.] acc={dist:.16f} |')
        
        pbar.update(1)
    

    if i == its:
        pbar.close()
        print('! Max iterations reached. Consider increasing max_iter or decreasing epsilon.')


    Z_i = Z_i_(chi, d, allowed_alife_neighbours, attr_size_graph, attractor_graph,homogenous_point_attractor,require_full_rattling,all_rattlers)

    observed = observables.calc_marginals(chi)

    Phi_RS = np.log(Z_i) - d / 2 * np.log(observed['Z_ij'])

    entropy = Phi_RS + observed['Legendre']


    return {
        **observed,
        **config,
        'init_chi': init_chi,
        'Phi_RS': Phi_RS,
        'entropy': entropy,
        'chi': chi,
        'converged': converged,
        'accuracy': abs(chi_old - chi__).sum()
    }
