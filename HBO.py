import math
import random
import numpy as np

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

        self.heap = Heap(N, N, self.D, self.QAP_fit, use_raw_perm)
        self.heap.build_heap()

    def process_random_keys(self, probs):
        # turn probs into perms
        return [sorted(probs).index(x) for x in probs]
    
    def eval_QAP(self, distances, flows, perm):
        # perm - indices - locations, values - facilities

        total = 0.0
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

            for I in range(self.N-1, 1, -1):
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

        return self.heap[0].key, self.process_random_keys(self.heap[0].value)

def main():
    # distances = [
    #     [0, 2, 3],
    #     [2, 0, 4],
    #     [3, 4, 0]
    # ]
    # flows = [
    #         [0, 5, 6],
    #         [5, 0, 1],
    #         [6, 1, 0]
    #     ]

    # -----------
    # Chr 12a
    # D = 12
    # distances = [
    #     [0, 90, 10, 23, 43, 0, 0, 0, 0, 0, 0, 0],
    #     [90, 0, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0],
    #     [10, 0, 0, 0, 0, 0, 26, 16, 0, 0, 0, 0],
    #     [23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 88, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 16, 0, 0, 0, 0, 0, 0, 96, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 29, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 37],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0]
    # ]


    # flows = [
    #     [0, 36, 54, 26, 59, 72, 9, 34, 79, 17, 46, 95],
    #     [36, 0, 73, 35, 90, 58, 30, 78, 35, 44, 79, 36],
    #     [54, 73, 0, 21, 10, 97, 58, 66, 69, 61, 54, 63],
    #     [26, 35, 21, 0, 93, 12, 46, 40, 37, 48, 68, 85],
    #     [59, 90, 10, 93, 0, 64, 5, 29, 76, 16, 5, 76],
    #     [72, 58, 97, 12, 64, 0, 96, 55, 38, 54, 0, 34],
    #     [9, 30, 58, 46, 5, 96, 0, 83, 35, 11, 56, 37],
    #     [34, 78, 66, 40, 29, 55, 83, 0, 44, 12, 15, 80],
    #     [79, 35, 69, 37, 76, 38, 35, 44, 0, 64, 39, 33],
    #     [17, 44, 61, 48, 16, 54, 11, 12, 64, 0, 70, 86],
    #     [46, 79, 54, 68, 5, 0, 56, 15, 39, 70, 0, 18],
    #     [95, 36, 63, 85, 76, 34, 37, 80, 33, 86, 18, 0]
    # ]

    # -----------
    # Chr18a
    # D = 18
    # opt = 11098
    # distances = [
    # [0,71,40,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # [71,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # [40,0,0,0,27,26,0,0,0,0,0,0,0,0,0,0,0,0],
    # [7,0,0,0,0,0,74,20,0,0,0,0,0,0,0,0,0,0],
    # [0,0,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    # [0,0,26,0,0,0,0,0,62,0,0,0,0,0,0,0,0,0],
    # [0,0,0,74,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    # [0,0,0,20,0,0,0,0,0,0,65,0,0,0,0,0,0,0],
    # [0,0,0,0,0,62,0,0,0,0,0,87,33,0,0,0,0,0],
    # [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    # [0,0,0,0,0,0,0,65,0,0,0,0,0,10,99,0,0,0],
    # [0,0,0,0,0,0,0,0,87,0,0,0,0,0,0,0,0,0],
    # [0,0,0,0,0,0,0,0,33,0,0,0,0,0,0,77,0,0],
    # [0,0,0,0,0,0,0,0,0,0,10,0,0,0,0,0,0,0],
    # [0,0,0,0,0,0,0,0,0,0,99,0,0,0,0,0,30,0],
    # [0,0,0,0,0,0,0,0,0,0,0,0,77,0,0,0,0,74],
    # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,0,0,0],
    # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,74,0,0]
    # ]

    # flows = [
    # [0,13,26,87,39,85,1,56,54,88,36,4,46,90,28,1,2,61],
    # [13,0,10,16,71,23,14,76,82,85,12,70,52,51,67,13,17,77],
    # [26,10,0,2,82,3,60,70,43,43,11,71,1,14,60,70,57,85],
    # [87,16,2,0,12,28,74,40,89,12,86,86,38,21,54,75,55,36],
    # [39,71,82,12,0,91,47,1,9,66,60,62,90,19,44,88,58,63],
    # [85,23,3,28,91,0,84,64,57,61,45,19,30,64,23,38,77,13],
    # [1,14,60,74,47,84,0,36,27,68,61,11,35,94,51,55,26,19],
    # [56,76,70,40,1,64,36,0,4,32,36,48,12,16,49,54,96,29],
    # [54,82,43,89,9,57,27,4,0,46,81,60,64,50,14,52,30,16],
    # [88,85,43,12,66,61,68,32,46,0,43,95,57,88,21,91,83,50],
    # [36,12,11,86,60,45,61,36,81,43,0,74,76,18,44,40,36,23],
    # [4,70,71,86,62,19,11,48,60,95,74,0,66,61,68,81,17,80],
    # [46,52,1,38,90,30,35,12,64,57,76,66,0,94,27,11,43,50],
    # [90,51,14,21,19,64,94,16,50,88,18,61,94,0,97,73,55,58],
    # [28,67,60,54,44,23,51,49,14,21,44,68,27,97,0,63,99,35],
    # [1,13,70,75,88,38,55,54,52,91,40,81,11,73,63,0,12,46],
    # [2,17,57,55,58,77,26,96,30,83,36,17,43,55,99,12,0,49],
    # [61,77,85,36,63,13,19,29,16,50,23,80,50,58,35,46,49,0]
    # ]

    # -----------
    # Chr25a
    # D = 25
    distances = [
    [0,8,5,14,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [8,0,0,0,0,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [5,0,0,0,0,0,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [14,0,0,0,0,0,0,2,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [12,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,23,0,0,0,0,0,0,0,0,8,13,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,2,0,0,0,0,0,0,0,0,29,35,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,4,0,0,0,0,0,0,0,0,0,12,15,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,13,0,0,0,0,0,0,0,0,0,0,7,9,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,12,0,0,0,0,0,0,0,0,10,3,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,0,20,35,5,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,26,31],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,0,0,0,0,0]
    ]

    flows = [
    [0,2,24,26,30,2,24,25,35,22,22,9,25,52,63,50,3,24,22,29,23,5,25,37,12],
    [2,0,22,20,35,22,4,5,29,22,35,44,25,60,22,35,24,22,29,30,5,5,7,19,22],
    [24,22,0,23,20,9,24,22,35,22,29,22,24,25,39,24,22,33,50,6,52,30,21,37,12],
    [26,20,23,0,29,5,24,36,39,25,34,3,20,20,35,22,6,52,63,22,20,35,36,39,17],
    [30,35,20,29,0,24,25,35,22,9,22,25,33,22,25,3,24,23,22,26,34,62,37,22,9],
    [2,22,9,5,24,0,35,22,25,24,29,33,6,50,82,25,22,33,24,29,23,26,12,25,14],
    [24,4,24,24,25,35,0,35,22,44,3,50,60,38,40,22,23,29,24,33,6,80,37,12,44],
    [25,5,22,36,35,22,35,0,24,22,25,29,36,40,3,50,22,35,32,23,5,24,3,15,12],
    [35,29,35,39,22,25,22,24,0,25,3,24,22,80,32,25,52,22,24,25,39,22,16,22,14],
    [22,22,22,25,9,24,44,22,25,0,40,29,32,60,25,28,33,8,9,22,24,25,15,12,40],
    [22,35,29,34,22,29,3,25,3,40,0,30,22,22,29,30,26,35,24,20,29,22,17,5,40],
    [9,44,22,3,25,33,50,29,24,29,30,0,22,33,64,92,25,25,29,30,20,50,50,29,14],
    [25,25,24,20,33,6,60,36,22,32,22,22,0,25,33,24,29,22,33,24,29,32,60,36,22],
    [52,60,25,20,22,50,38,40,80,60,22,33,25,0,29,35,50,44,32,24,23,28,38,40,80],
    [63,22,39,35,25,82,40,3,32,25,29,64,33,29,0,22,22,50,25,62,24,22,40,3,31],
    [50,35,24,22,3,25,22,50,25,28,30,92,24,35,22,0,29,23,50,24,29,25,12,50,27],
    [3,24,22,6,24,22,23,22,52,33,26,25,29,50,22,29,0,24,39,50,22,50,13,22,52],
    [24,22,33,52,23,33,29,35,22,8,35,25,22,44,50,23,24,0,29,32,24,25,19,37,12],
    [22,29,50,63,22,24,24,32,24,9,24,29,33,32,25,50,39,29,0,29,26,35,24,31,14],
    [29,30,6,22,26,29,33,23,25,22,20,30,24,24,62,24,50,32,29,0,28,24,33,23,27],
    [23,5,52,20,34,23,6,5,39,24,29,20,29,23,24,29,22,24,26,28,0,35,6,7,39],
    [5,5,30,35,62,26,80,24,22,25,22,50,32,28,22,25,50,25,35,24,35,0,80,14,12],
    [25,7,21,36,37,12,37,3,16,15,17,50,60,38,40,12,13,19,24,33,6,80,0,10,15],
    [37,19,37,39,22,25,12,15,22,12,5,29,36,40,3,50,22,37,31,23,7,14,10,0,25],
    [12,22,12,17,9,14,44,12,14,40,40,14,22,80,31,27,52,12,14,27,39,12,15,25,0]
    ]

    # for D = 12
    # N ~= 20 x D
    # T ~= 25 * N

    hbo = HBO(distances, flows, 200, True)
    q, perm = hbo.run(30000)
    print('---')
    print(q, perm)


if __name__ == "__main__":
    main()

