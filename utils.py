from collections import defaultdict

import numpy as np
import networkx as nx


def get_G_from_edges(edges):
    Graph = nx.Graph()
    for edge in edges:
        Graph.add_edge(edge[0], edge[1])
    return Graph

def load_training_data(f_name):
    print('We are loading training data from:', f_name)
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split('\t')
            x, y = int(words[0]), int(words[1])
            all_edges.append((x, y))
            all_edges.append((y, x))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    all_edges = list(set(all_edges))
    print('total training nodes: ' + str(len(all_nodes)))
    print('Finish loading training data')
    return all_edges, all_nodes


def load_testing_data(f_name):
    print('We are loading testing data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split('\t')
            x, y = int(words[1]), int(words[2])
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Finish loading testing data')
    return true_edge_data_by_type, false_edge_data_by_type

def load_node_type(f_name):
    print('We are loading node type from:', f_name)
    node_type = {}
    type_nodes = defaultdict(list)
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split('\t')
            node_type[int(items[0])] = items[1]
    node_types = list(set(node_type.values()))
    type_vocab = dict(zip(node_types, list(range(len(node_types)))))
    for node in node_type.keys():
        node_type[node] = type_vocab[node_type[node]]
        type_nodes[node_type[node]].append(node)
    return node_type, type_vocab, type_nodes

def load_network_schemas(f_name, type_vocab):
    print('We are loading network schemas from:', f_name)
    schemas = []
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split('-')
            schema = [type_vocab[item] for item in items]
            schemas.append(schema)
    return schemas

def build_neighbor(node_types, edges):
    type_neighbors = defaultdict(list)
    for edge in edges:
        type_neighbors[edge[0]].append(node_types[edge[1]])
        type_neighbors[edge[1]].append(node_types[edge[0]])
    for node in type_neighbors.keys():
        type_neighbors[node] = list(set(type_neighbors[node]))
    return type_neighbors