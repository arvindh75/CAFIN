import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import pickle

import networkx as nx
from torch_geometric.utils import to_networkx
from utils import *

parser = argparse.ArgumentParser(description="Train GraphSAGE for link prediction")
parser.add_argument("--dataset", type=str, help="The dataset", default="Cora")
args = parser.parse_args()
data = get_data(args.dataset)

with open(f"./data/{args.dataset}/edge_index.pkl", "rb") as f:
    edge_index = pickle.load(f)

data.edge_index = edge_index

G = to_networkx(data, to_undirected=True)
print(nx.info(G))

NODES = len(G.nodes)
MAX_VAL = len(G.nodes) - 1

print(f"Starting Distance calculation")
dist = dict(nx.all_pairs_shortest_path_length(G))
print(f"Finished Distance calculation\n", flush=True)

for i in range(NODES):
    if i not in dist:
        dist[i] = {}
    for j in range(NODES):
        if i == j:
            dist[i][j] = 1
        elif j not in dist[i] and i != j:
            dist[i][j] = MAX_VAL

with open(f"./data/{args.dataset}/dist.pkl", "wb") as f:
    pickle.dump(dist, f)
