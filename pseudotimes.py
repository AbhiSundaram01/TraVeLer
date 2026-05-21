import numpy as np
import torch
import networkx as nx
from neural_k_forms.chains import generate_integration_matrix
from graph_utils import get_all_edges

def get_trees(A, P, root_cell):
    G = nx.from_numpy_array(A.detach().numpy(), create_using=nx.DiGraph)
    trees = []
    for root_cluster in range(P.shape[1]):
        G_copy = G.copy()
        prob = P[root_cell][root_cluster]
        for u in list(G_copy.predecessors(root_cluster)):
            G_copy.remove_edge(u, root_cluster)
        for n in G_copy.nodes:
            print(n, "in-degree:", G_copy.in_degree(n), "out-degree:", G_copy.out_degree(n))
        msa = nx.maximum_spanning_arborescence(G_copy)
        trees.append((prob, msa))
    
    return trees
def get_path_from_root(tree, root, target):
    msa = tree
    path = [target]
    current = target
    
    while current != root:
        preds = list(msa.predecessors(current))
        
        if not preds:
            raise ValueError("Target not reachable from root")
        
        current = preds[0]
        path.append(current)
    
    path.reverse()
    return path
def get_chain_from_path(path, x_out):
    chain = torch.zeros((len(path) - 1, 2, x_out.shape[1]), device=x_out.device)
    for idx, _ in enumerate(path):
        if idx != len(path) -1:
            chain[idx, 0] = x_out[path[idx]]
            chain[idx, 1] = x_out[path[idx+1]]
    return chain

def get_edge_info(x_out, adj_out, vf):
    chain, _ = get_all_edges(x_out, adj_out)
    edge_flows = generate_integration_matrix(vf, chain)
    edge_list = [(i, j) for i in range(adj_out.shape[0]) for j in range(adj_out.shape[1]) if i != j]
    edge_map = { (i, j): idx for idx, (i, j) in enumerate(edge_list) }
    return edge_flows, edge_map

def get_flow(trees, target_cell, P, x_out, vf, edge_flows, edge_map):
    flow_over_trees = 0.0
    for prob, tree in trees:
        flow_over_paths = 0.0
        for target, cluster_prob in enumerate(P[target_cell, :]):
            root = [n for n in tree.nodes if tree.in_degree(n) == 0]
            root = root[0]
            if root == target:
                continue
            path = get_path_from_root(tree, root, target)
            flow = 0
            counter = 0  
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                edge_idx = edge_map[(u, v)]
                flow += edge_flows[edge_idx]
                counter +=1
            if counter != 0:
                flow = flow / counter
            flow_over_paths += flow * cluster_prob
        flow_over_trees += flow_over_paths * prob
    return flow_over_trees
    





