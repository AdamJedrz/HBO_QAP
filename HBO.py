import math
import random
import numpy as np
import time
from pathlib import Path
import statistics

class Node:
    def __init__(self, key, value):
        self.key = key # fitness
        self.value = value # coords

    def __str__(self):
        return f"{self.key}, {self.value}"

class Heap:
    def __init__(self, P, N, D, f, use_raw_perms):
        # P - population of search agents
        # N - population size
        # f - fitness function
        self.d = 3 #d-ary heap
        self.arr = [None for _ in range(N)]

        self.P = P # idk what this is for
        self.N = N
        self.D = D
        self.f = f
        self.use_raw_perms = use_raw_perms

    def __getitem__(self, key):
        return self.arr[key]
    
    def __str__(self):
        return str([str(x) for x in self.arr])

    def parent(self, i):
        # returns parent of a node with index i
        return math.floor((i-1)/self.d)
    
    def child(self, i, j):
        # returns the j-th child of node with index i
        return self.d*i - self.d + j + 4

    def depth(self, i):
        #returns the depth at which node with index i is located
        return math.ceil(math.log((i + 1) * (self.d - 1) + 1, self.d)) - 1

    def colleague(self, i):
        # returns the index of a random colleague of a node with index i
        depth = self.depth(i)

        while True:
            ci = random.randint(int((math.pow(self.d, depth) - 1)/(self.d - 1)), int((math.pow(self.d, depth+1) - 1)/(self.d - 1) - 1))
            if ci < self.N:
                break

        return ci

    def heapify_up(self, i):
        # search upwards and insert node i at its correct location
        
        parent = self.parent(i)
        while i != 0 and self[i].key < self[parent].key:
            self.arr[i], self.arr[parent] = self.arr[parent], self.arr[i]
            i = parent
            parent = self.parent(i)

    def build_heap(self):
        # P - population of search agents
        # N - population size

        for i in range(0, self.N):
            if not self.use_raw_perms:
                self.arr[i] = Node(0, [random.random() for _ in range(self.D)])

            else:
                n = Node(0, [-1 for _ in range(self.D)])
                for j in range(self.D):
                    while True:
                        r = random.randint(0, self.D-1)
                        
                        if r not in n.value:
                            break
                    
                    n.value[j] = r

                self.arr[i] = n

            self.arr[i].key = self.f(self[i].value)
            self.heapify_up(i)


class HBO:
    def __init__(self, distances, flows, N, use_raw_perm=False, pmx_prob=1.0):
        self.distances = distances
        self.flows = flows

        self.N = N
        self.D = len(distances)
        self.f = self.QAP_fit
        self.use_raw_perm = use_raw_perm
        self.pmx_prob = pmx_prob

        self.C_div_constant = 25

        self.treat_as_upper_triang = self.is_triang(self.distances) and self.is_triang(flows)
        
        self.heap = Heap(N, N, self.D, self.QAP_fit, use_raw_perm)
        self.heap.build_heap()

    def is_triang(self, X):
        is_symmetrical = True
        is_triang = True

        for x in range(len(X)):
            for y in range(x, len(X)):
                if X[x][y] != X[y][x]:
                    is_symmetrical = False
                
                if X[y][x] != 0:
                    is_triang = False

        return is_symmetrical or is_triang

    def process_random_keys(self, probs):
        # turn probs into perms
        return [sorted(probs).index(x) for x in probs]
    
    def eval_QAP(self, distances, flows, perm):
        # perm - indices - locations, values - facilities

        total = 0.0
        if self.treat_as_upper_triang:
            for i1 in range(self.D):
                for i2 in range(i1, self.D):
                    total += distances[i1][i2] * flows[perm[i1]][perm[i2]]
            total *= 2

        else:
            for i1 in range(self.D):
                for i2 in range(self.D):
                    total += distances[i1][i2] * flows[perm[i1]][perm[i2]]

        return total
    
    def QAP_fit(self, x):
        if not self.use_raw_perm:
            perm = self.process_random_keys(x)

        else:
            perm = x

        return self.eval_QAP(self.distances, self.flows, perm)

    def gamma_eq(self, t, T):
        C = math.floor(T/self.C_div_constant)
        return abs(2 - ((t % (T/C)) / (T/(4*C))))

    def prob_bound_eq(self, t, T):
        p1 = 1 - (t/T)
        p2 = p1 + (1 - p1)/2

        return p1, p2
    
    def agent_update_rk(self, xi, x_temp, bi, B, ci, S, p1, p2, gamma):
        for k in range(self.D):
                    p = random.random()
                    lbd = 2 * random.random() - 1
                    
                    if p <= p1:
                        x_temp.value[k] = xi.value[k]

                    elif p > p1 and p <= p2:
                        x_temp.value[k] = B[k] + gamma * lbd * abs(B[k] - xi.value[k])
                    
                    elif p > p2 and ci.key < xi.key:
                        x_temp.value[k] = S[k] + gamma * lbd * abs(S[k] - xi.value[k])

                    else:
                        x_temp.value[k] = xi.value[k] + gamma * lbd * abs(S[k] - xi.value[k])

    def pmx(self, parent1, parent2, cut1, cut2):
        offspring = [0] * len(parent1)

        # Copy the middle section from `parent1`
        offspring[cut1:cut2] = parent1[cut1:cut2]

        # Copy the rest from `parent2`, resolving conflicts
        for i in (*range(0,cut1), *range(cut2,len(parent1))):
            candidate = parent2[i]
            while candidate in parent1[cut1:cut2]:  # handle successive mappings
                candidate = parent2[parent1.index(candidate)]
            offspring[i] = candidate

        return offspring
    
    def ox(self, perm1, perm2, c):
        c = self.D - c
        ls = perm1[:c]
        rs = perm2[c:]

        for i, x in enumerate(rs):
            if x in ls:
                for y in perm2:
                    if y not in rs and y not in ls:
                        rs[i] = y

        r = ls + rs
        return r

    def mut_swap(self, perm, n):
        # random swap
        for _ in range(n):
            idx1 = random.randint(0, self.D-1)
            while True:
                idx2 = random.randint(0, self.D-1)
                if idx1 != idx2:
                    break

            perm[idx1], perm[idx2] = perm[idx2], perm[idx1]

        return perm
    
    def mut_rot(self, perm, n):
        # shift in a random direction
        dir = random.randint(0, 1)
        b1 = n % self.D
        b2 = (self.D - n) % self.D

        if dir: #right
            perm = perm[-b1:] + perm[:b2]

        else: #left
            perm = perm[-b2:] + perm[:b1]

        return perm
    
    def mutate(self, perm, n):
        p = random.random()

        if p <= 0.5:
            return self.mut_swap(perm, n)
        
        else:
            return self.mut_rot(perm, n)
    
    def perm_dist(self, perm1, perm2):
        return sum([perm1[i] != perm2[i] for i in range(self.D)])//2
    
    def cross_perms(self, perm1, perm2, a):
        # pick random parts (length D*a) of the perms and do a crossover between them
        seq_len = round((self.D - 1) * a)
        sidx = random.randint(0, self.D - seq_len)
        eidx = sidx + seq_len

        if seq_len > 2:
            c1 = random.randint(0, seq_len-2)
            c2 = random.randint(c1+1, seq_len-1)
        
        else:
            c1 = 0
            c2 = max(seq_len-1, 0)

        rperm1 = perm1[sidx:eidx]
        rperm2 = perm2[sidx:eidx]

        for i, x in enumerate(rperm2):
              if x in perm1[:sidx] or x in perm1[eidx:]:
                    for elem in perm2:
                          if elem not in rperm2 and elem not in perm1[:sidx] and elem not in perm1[eidx:]:
                                rperm2[i] = elem

        p = random.random()
        if p <= self.pmx_prob:
            r = self.pmx(rperm1, rperm2, c1, c2)
            
        elif p > self.pmx_prob and p <= 1.0:
            r = self.ox(rperm1, rperm2, c1)

        return perm1[:sidx] + r + perm1[eidx:]

    def agent_update_perm(self, xi, x_temp, bi, B, ci, S, p1, p2, gamma):
        dir_p = random.random()
        op_p = random.random()

        c1 = random.randint(0, self.D-1)
        c2 = random.randint(c1+1, self.D)

        lbd = 2 * random.random() - 1
        gamma = gamma/2 # gamma = [-2, 2], scaled to [-1, 1]

        if dir_p <= p1:
            x_temp.value = xi.value.copy()
        
        elif dir_p > p1 and dir_p <= p2:
            x_temp.value = self.cross_perms(xi.value, B, abs(gamma*lbd))

        elif dir_p > p2 and ci.key < xi.key:
            x_temp.value = self.cross_perms(xi.value, S, abs(gamma*lbd))

        else:
            # no better colleague - mutate
            x_temp.value = self.mutate(xi.value.copy(), abs(round(gamma*lbd*self.perm_dist(xi.value, S)))) # oryginalny wsp. kowalewskiego K=2
        
    def run(self, T):
        self.C_div_constant = min(T, self.C_div_constant)

        for t in range(1, T+1):
            gamma = self.gamma_eq(t, T)
            p1, p2 = self.prob_bound_eq(t, T)

            for I in range(self.N-1, 0, -1):
                xi = self.heap[I]
                bi = self.heap[self.heap.parent(I)]
                ci = self.heap[self.heap.colleague(I)]

                B = bi.value
                S = ci.value

                x_temp = Node(xi.key, [0 for _ in range(self.D)])
                
                if not self.use_raw_perm:
                    self.agent_update_rk(xi, x_temp, bi, B, ci, S, p1, p2, gamma)

                else:
                    self.agent_update_perm(xi, x_temp, bi, B, ci, S, p1, p2, gamma)

                x_temp.key = self.f(x_temp.value)

                if x_temp.key < xi.key:
                    self.heap.arr[I] = x_temp

                self.heap.heapify_up(I)

            if t % (T//100) == 0:
                print(f"{t//(T//100)}%")

        if self.use_raw_perm:
            return self.heap[0].key, self.heap[0].value.copy()
        else:
            return self.heap[0].key, self.process_random_keys(self.heap[0].value)
        

## Data loading

def _read_all_ints(path):
    path = Path(path)
    return [int(x) for x in path.read_text(encoding="utf-8", errors="ignore").split()]


def read_qap_dat(path):

    nums = _read_all_ints(path)
    if not nums:
        raise ValueError(f"Empty file: {path}")

    n = nums[0]
    expected = 1 + 2 * n * n
    if len(nums) < expected:
        raise ValueError(f"Too little data in file {path}. Expected {expected} numbers, got {len(nums)}")

    A = np.array(nums[1:1 + n * n], dtype=np.int64).reshape(n, n)
    B = np.array(nums[1 + n * n:1 + 2 * n * n], dtype=np.int64).reshape(n, n)

    return {
        "name": Path(path).stem.lower(),
        "n": n,
        "A": A,
        "B": B,
    }


def invert_qaplib_perm(perm_facility_to_location_1based):
    """
    QAPLIB:
        facility -> location   (1-based)
    Internal representation:
        location -> facility   (0-based)

    """

    n = len(perm_facility_to_location_1based)
    loc_to_fac = [-1] * n

    for facility_idx, loc_1based in enumerate(perm_facility_to_location_1based):
        loc_0based = loc_1based - 1
        loc_to_fac[loc_0based] = facility_idx

    return loc_to_fac


def read_qap_solution(path):

    nums = _read_all_ints(path)
    if len(nums) < 2:
        raise ValueError(f"Solution file {path} is too short")

    n = nums[0]
    objective = nums[1]

    perm_qaplib_1based = None
    perm_internal_0based = None

    if len(nums) >= 2 + n:
        perm_qaplib_1based = nums[2:2 + n]
        perm_internal_0based = invert_qaplib_perm(perm_qaplib_1based)

    return {
        "name": Path(path).stem.lower(),
        "n": n,
        "objective": objective,
        "perm_qaplib_1based": perm_qaplib_1based,
        "perm_internal_0based": perm_internal_0based,
    }


def load_instance(instance_name, qapdata_dir="qapdata", qapsoln_dir="qapsoln"):

    instance_name = instance_name.lower()

    dat_path = Path(qapdata_dir) / f"{instance_name}.dat"
    if not dat_path.exists():
        raise FileNotFoundError(f"Instance file not found: {dat_path}")

    instance = read_qap_dat(dat_path)

    solution = None
    for ext in [".sln", ".dat", ".txt"]:
        sol_path = Path(qapsoln_dir) / f"{instance_name}{ext}"
        if sol_path.exists():
            solution = read_qap_solution(sol_path)
            break

    return {
        "instance": instance,
        "solution": solution,
    }


def load_family(prefix, qapdata_dir="qapdata", qapsoln_dir="qapsoln", limit=None):
    """
    Loads several instances from one family, for example:
        load_family("chr")

    limit - optional number of first instances to load
    """
    prefix = prefix.lower()
    data_dir = Path(qapdata_dir)

    dat_files = sorted([p for p in data_dir.glob("*.dat") if p.stem.lower().startswith(prefix)])
    if limit is not None:
        dat_files = dat_files[:limit]

    result = []
    for dat_file in dat_files:
        name = dat_file.stem.lower()
        item = load_instance(name, qapdata_dir=qapdata_dir, qapsoln_dir=qapsoln_dir)
        result.append(item)

    return result


## Evaluation

def gap_percent(value, reference):
    return 100.0 * (value - reference) / reference


def run_hbo_once(instance_item, N, T, use_raw_perm=True, seed=None):
    """
    Runs HBO once on a single instance.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    A = instance_item["instance"]["A"].tolist()
    B = instance_item["instance"]["B"].tolist()

    start = time.perf_counter()
    hbo = HBO(A, B, N, use_raw_perm=use_raw_perm)
    best_cost, best_perm = hbo.run(T)
    elapsed = time.perf_counter() - start

    return {
        "best_cost": best_cost,
        "best_perm": best_perm,
        "time_sec": elapsed,
    }


def evaluate_instance(instance_item, N, T, n_runs=10, use_raw_perm=True, seed0=123):
    """
    Multiple runs on a single instance.
    """
    runs = []
    for r in range(n_runs):
        out = run_hbo_once(instance_item, N=N, T=T, use_raw_perm=use_raw_perm, seed=seed0 + r)
        runs.append(out)

    costs = [r["best_cost"] for r in runs]
    times = [r["time_sec"] for r in runs]

    summary = {
        "name": instance_item["instance"]["name"],
        "n": instance_item["instance"]["n"],
        "runs": n_runs,
        "best_cost": min(costs),
        "mean_cost": statistics.mean(costs),
        "median_cost": statistics.median(costs),
        "std_cost": statistics.pstdev(costs) if len(costs) > 1 else 0.0,
        "worst_cost": max(costs),
        "mean_time_sec": statistics.mean(times),
        "all_costs": costs,
        "all_times": times,
    }

    solution = instance_item["solution"]
    if solution is not None:
        ref = solution["objective"]
        summary["reference_cost"] = ref
        summary["best_gap_pct"] = gap_percent(summary["best_cost"], ref)
        summary["mean_gap_pct"] = gap_percent(summary["mean_cost"], ref)
        summary["match_reference_count"] = sum(c == ref for c in costs)
        summary["match_reference_rate"] = summary["match_reference_count"] / n_runs

    return summary


def evaluate_family_items(items, N, T, n_runs=10, use_raw_perm=True, seed0=123, verbose=True):
    """
    Evaluation of multiple instances from one family.
    """
    results = []
    for i, item in enumerate(items):
        summary = evaluate_instance(
            item,
            N=N,
            T=T,
            n_runs=n_runs,
            use_raw_perm=use_raw_perm,
            seed0=seed0 + 1000 * i,
        )
        results.append(summary)

        if verbose:
            msg = f'{summary["name"]}: best={summary["best_cost"]}, mean={summary["mean_cost"]:.2f}'
            if "reference_cost" in summary:
                msg += f', ref={summary["reference_cost"]}, best_gap={summary["best_gap_pct"]:.3f}%'
            print(msg)

    return results



def main():
    #item = load_instance("chr12a", qapdata_dir="qapdata", qapsoln_dir="qapsoln")

    #summary = evaluate_instance(
    #    item,
    #    N=120,
    #    T=6000,
    #    n_runs=3,
    #    use_raw_perm=True,
    #    seed0=123
    #)

    #print(summary)


    items = load_family("chr", qapdata_dir="qapdata", qapsoln_dir="qapsoln", limit=2)

    results = evaluate_family_items(
        items,
        N=120,
        T=6000,
        n_runs=1,
        use_raw_perm=False,
        seed0=123
    )

    print(results)

if __name__ == "__main__":
    main()