from collections import defaultdict, Counter
import random
import gc

import numpy as np


class SNSampler():
    def __init__(self, Ori_Graph, Samp_Graph,  context, node_types, num_walks, window_size, type_walk_nums=None, type_vocabs=None, type_neighbor=None, coar_type_neighbor=None):
        self.Ori_Graph = Ori_Graph
        self.Samp_Graph = Samp_Graph
        self.node_types = node_types
        self.num_walks = num_walks
        self.window_size = window_size
        self.context = context
        self.type_walk_nums = type_walk_nums
        self.type_vocabs = type_vocabs
        self.type_neighbor = type_neighbor
        self.coar_type_neighbor = coar_type_neighbor
        if type_neighbor != None:
            self.typed_Graph = self.build_typed_Graph(Samp_Graph)
            self.Ori_typed_Graph = self.build_typed_Graph(Ori_Graph)
        self.degree_prob = {}
        self.import_prob = {}
        if len(self.context) == 0:
            self.build_degree_prob()
        else:
            self.build_import_prob()

    def build_typed_Graph(self, Graph):
        typed_Graph = {}
        for node_type in self.type_vocabs.values():
            typed_Graph[node_type] = defaultdict(list)
        for node in Graph.nodes():
            for tail in Graph.neighbors(node):
                typed_Graph[self.node_types[tail]][node].append(tail)
        return typed_Graph

    def het_sampled_walk(self, start):
        rand = random.Random()
        walk = [start]
        while len(walk) < self.window_size+1:
            cur = walk[-1]
            candi_type = rand.choice(self.coar_type_neighbor[cur])
            neighbors = self.typed_Graph[candi_type][cur]
            candidates = neighbors
            if candidates:
                candi = rand.choice(candidates)
                if candi == start:
                    candi_type = rand.choice(self.coar_type_neighbor[start])
                    walk.append(rand.choice(self.typed_Graph[candi_type][start]))
                else:
                    walk.append(candi)
            else:
                candi_type = rand.choice(self.coar_type_neighbor[start])
                walk.append(rand.choice(self.typed_Graph[candi_type][start]))
                self.context.append(walk[-1])
        return [(start, node) for node in walk[1:]]

    def het_walk(self, start):
        rand = random.Random()
        walk = [start]
        while len(walk) < self.window_size+1:
            cur = walk[-1]
            candi_type = rand.choice(self.type_neighbor[cur])
            neighbors = self.Ori_typed_Graph[candi_type][cur]
            candidates = neighbors
            if candidates:
                candi = rand.choice(candidates)
                if candi == start:
                    candi_type = rand.choice(self.type_neighbor[start])
                    walk.append(rand.choice(self.Ori_typed_Graph[candi_type][start]))
                else:
                    walk.append(candi)
            else:
                candi_type = rand.choice(self.type_neighbor[start])
                walk.append(rand.choice(self.Ori_typed_Graph[candi_type][start]))
                self.context.append(walk[-1])
        return [(start, node) for node in walk[1:]]

    def sn_sample(self, Graph, Ori_candis):
        node_pairs = []
        sampled_nodes = list(Graph.nodes())
        ori_nodes = Ori_candis
        if len(self.import_prob) == 0:
            for node in sampled_nodes:
                self.context.append(node)
                for _ in range(self.degree_prob[node]):
                    node_pairs += self.het_sampled_walk(node)
            for node in ori_nodes:
                    self.context.append(node)
                    for _ in range(self.degree_prob[node]):
                        node_pairs += self.het_walk(node)
        else:
            for node in sampled_nodes:
                self.context.append(node)
                for _ in range(self.import_prob[node]):
                    node_pairs += self.het_sampled_walk(node)
            for node in ori_nodes:
                self.context.append(node)
                for _ in range(self.import_prob[node]):
                    node_pairs += self.het_walk(node)
        context = self.context
        return node_pairs, context

    def sn_sample_multi(self, nodes, Ori_candis):
        self.context = []
        node_pairs = []
        ori_nodes = Ori_candis
        if self.type_neighbor == None:
            if len(self.import_prob) == 0:
                for node in nodes:
                    self.context.append(node)
                    for _ in range(self.degree_prob[node]):
                        node_pairs += self.walk(node)
            else:
                for node in nodes:
                    self.context.append(node)
                    for _ in range(self.import_prob[node]):
                        node_pairs += self.walk(node)
        else:
            if len(self.import_prob) == 0:
                for node in nodes:
                    self.context.append(node)
                    for _ in range(self.degree_prob[node]):
                        node_pairs += self.het_sampled_walk(node)
                for node in ori_nodes:
                    self.context.append(node)
                    for _ in range(self.degree_prob[node]):
                        node_pairs += self.het_walk(node)
            else:
                for node in nodes:
                    self.context.append(node)
                    for _ in range(self.import_prob[node]):
                        node_pairs += self.het_sampled_walk(node)
                for node in ori_nodes:
                    self.context.append(node)
                    for _ in range(self.import_prob[node]):
                        node_pairs += self.het_walk(node)
        context = self.context
        gc.collect()
        return node_pairs, context

    def build_degree_prob(self, ):
        edge_size = len(self.Ori_Graph.edges())
        node_size = len(self.Ori_Graph.nodes())
        for node in self.Ori_Graph.nodes():
            self.degree_prob[node] = int(np.ceil((self.Ori_Graph.degree(node) / (2*edge_size)) * (self.num_walks*node_size)))

    def build_import_prob(self, ):
        node_size = len(self.Ori_Graph.nodes())
        import_dic = Counter(self.context)
        for node in import_dic.keys():
            self.import_prob[node] = int(np.ceil(import_dic[node] / len(self.context) * (self.num_walks*node_size)))