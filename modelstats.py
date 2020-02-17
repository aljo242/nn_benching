import argparse
import torch
from torch.hub import load_state_dict_from_url
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
INFERENCE_SIZE = (244, 244)

def get_model_graph(model):
    dummy_input = torch.randn(1, 3, INFERENCE_SIZE[0], INFERENCE_SIZE[1])
    print("Get model graph")
    dummy_output = model(dummy_input)
    params = model.state_dict()
    param_map = {id(v): k for k, v in params.items()}

    seen = set()

    nodes = {}

    edges = {}

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_edge(f, t):
        fromid = str(id(f))
        if fromid in edges:
            edges[fromid].append(str(id(t)))
        else:
            edges[fromid] = [str(id(t))]

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                nodes[str(id(var))] = size_to_str(var.size())
            elif hasattr(var, 'variable'):
                subvar = var.variable
                node_name = '%s\n %s' % (param_map.get(id(subvar)), size_to_str(subvar.size()))
                nodes[str(id(var))] = node_name
            else:
                nodes[str(id(var))] = str(type(var).__name__)
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        add_edge(u[0], var)
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    add_edge(t, var)
                    add_nodes(t)
    add_nodes(dummy_output.grad_fn)
    print("Model graph complete")
    return nodes, edges

def __topo_sort(edge_lookup, nodeid, visited, stack):
    visited[nodeid] = True
    node_edges = edge_lookup[nodeid] if nodeid in edge_lookup else []
    for node in node_edges:
        if node not in visited:
            __topo_sort(edge_lookup, node, visited, stack)
        stack.insert(0, nodeid)

def topological_sort(e):
    print("Sort edge map")
    visited = {}
    stack = []
    for idx, nodeid in enumerate(e.keys()):
        if nodeid not in visited:
            __topo_sort(e, nodeid, visited, stack)
    print("Edge map sorted")
    return stack

def get_critical_path(model):
    nodes, edges = get_model_graph(model)
    from_nodes_sorted = topological_sort(edges)
    latencies = {}

    for node in from_nodes_sorted:
        e = edges[node] if node in edges  else []
        if node not in latencies:
            latencies[node] = 0
        for edge in e:
            edge_latency = latencies[edge] if edge in latencies else 0
            latencies[edge] = max(edge_latency, latencies[node] + 1)
    return max(latencies.values()), latencies, [latencies[k] for k in from_nodes_sorted]


