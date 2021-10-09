import random

from collections import defaultdict
import numpy as np


class GraphCoarsener():
    def __init__(self, node_types, type_nodes, coarsen_rate=0.5):
        """
        coarsen_rate : rate of remaining nodes
        """
        self.node_types = node_types
        self.type_nodes = type_nodes
        self.coarsen_rate = coarsen_rate
        print("finish Coarsener init")

    def coarsen(self, Graph, node_importance=None, center_type=2):
        candidates = self.typed_candi_sampling(node_importance, center_type)
        coarsened_graph = self.graph_coarsen(Graph, candidates)
        return coarsened_graph, candidates

    def typed_candi_sampling(self, node_importance, center_type=2):
        candidates = []
        typed_importance = defaultdict(list)
        for node in node_importance:
            typed_importance[self.node_types[node]].append(node)
        for node_type in typed_importance.keys():
            if node_type != center_type:
                candidates += typed_importance[node_type][:int(np.ceil(len(typed_importance[node_type])*self.coarsen_rate))]
            else:
                pass
        candidates = list(set(candidates))
        return candidates

    def graph_coarsen(self, Graph, candidates):
        rand = random.Random()
        new_Graph = Graph.copy(as_view=False)
        deg_rank = {}
        for node in Graph.nodes():
            deg_rank[node] = Graph.degree(node)
        tmp = sorted(deg_rank.items(), key=lambda x: x[1], reverse=True)
        rm_rank = {}
        
        for node in candidates:
            rm_rank[node] = Graph.degree(node)
            neighbors = list(new_Graph.neighbors(node))
            neighbor_size = len(neighbors)

            if neighbor_size > 1:
                for i in range(neighbor_size-1):
                    new_Graph.add_edge(neighbors[i], rand.choice(neighbors[i+1:]))
                new_Graph.remove_node(node)
            else:
                new_Graph.remove_node(node)
        print('ori_graph nodes: {}'.format(len(list(Graph.nodes()))))
        print('coar_graph nodes: {}'.format(len(list(new_Graph.nodes()))))
        return new_Graph