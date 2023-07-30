import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import json

import networkx as nx
from networkx.algorithms.centrality import (
    degree_centrality,
    eigenvector_centrality,
)
from torch_geometric.utils import to_networkx
from utils import *

parser = argparse.ArgumentParser(description="Train GraphSAGE for node classification")
parser.add_argument("--dataset", type=str, help="The dataset", default="Cora")
args = parser.parse_args()
data = get_data(args.dataset)

G = to_networkx(data, to_undirected=True)  #
print(nx.info(G))

MAX_VAL = len(G.nodes) - 1

print(f"Starting Centrality calculation", flush=True)
degc_dict = degree_centrality(G)
eigc_dict = eigenvector_centrality(G)
print(f"Finished Centrality calculation\n", flush=True)

with open(f"./data/{args.dataset}/deg_c.json", "w") as f:
    json.dump(degc_dict, f)

with open(f"./data/{args.dataset}/eig_c.json", "w") as f:
    json.dump(eigc_dict, f)
