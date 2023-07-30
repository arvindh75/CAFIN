import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import pickle
import random
from collections import defaultdict

import networkx as nx
from torch_geometric.utils import to_networkx
from utils import *


def find_landmarks(subG: nx.Graph(), method="random"):
    nodes = [x for x in subG.nodes()]
    cnt_landmarks = min(len(nodes), 100)
    if method == "random":
        return random.sample(nodes, cnt_landmarks)
    elif method == "degree":
        nodes.sort(reverse=True, key=lambda x: subG.degree[x])
        l = nodes[:cnt_landmarks]
        return l
    else:
        raise ValueError("Method doesn't match in find_landmarks")


# Takes Subgraph and it's index computes distance of each node from landmarks
def preprocess_distances(g: nx.Graph(), dataset: str, method="random"):
    directory = f"./data/{dataset}/distances/landmark_{method}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    js = {
        "landmarks": [],
        "comp_id": defaultdict(int),
        "edges": set(),
        "distances": defaultdict(list),
    }

    for id, subG in enumerate(g):
        # Assigning component id to nodes
        nodes = [x for x in subG.nodes()]
        for x in nodes:
            js["comp_id"][x] = id

        # Edges have distance 1
        for u, v in subG.edges():
            js["edges"].add((u, v))
            js["edges"].add((v, u))

        # Find landmarks
        L = find_landmarks(subG, method)
        js["landmarks"].append(L)

        # compute distance of each node in this subgraph from landmarks
        # js['distances'][n] => L length array
        for l in L:
            length = nx.single_source_shortest_path_length(subG, l)
            for n in nodes:
                js["distances"][n].append(length[n])

    with open(directory + f"data.pkl", "wb") as f:
        pickle.dump(js, f)

    return js


# Returns distance between node u and v.
def compute_distance(u: int, v: int, dc: dict):
    if u == v:
        return 1
    INF = 10**5
    if dc["comp_id"][u] != dc["comp_id"][v]:
        return INF
    elif (u, v) in dc["edges"]:
        return 1
    else:
        # tightest upperbound approach
        mn = INF
        for a, b in zip(dc["distances"][u], dc["distances"][v]):
            mn = min(mn, a + b)
        return mn


parser = argparse.ArgumentParser(description="Train GraphSAGE for link prediction")
parser.add_argument("--dataset", type=str, help="The dataset", default="Cora")
args = parser.parse_args()
data = get_data(args.dataset)

with open(f"./data/{args.dataset}/edge_index.pkl", "rb") as f:
    edge_index = pickle.load(f)

data.edge_index = edge_index

G = to_networkx(data, to_undirected=True)

components = list(nx.connected_components(G))
subGs = [G.subgraph(i).copy() for i in components]
js = preprocess_distances(subGs, args.dataset, method="random")
