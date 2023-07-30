import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import pickle
import random

import torch
from torch_geometric.utils import to_networkx
from utils import *

parser = argparse.ArgumentParser(description="Sample Subgraphs")
parser.add_argument("--dataset", type=str, help="The dataset", default="Cora")
args = parser.parse_args()
dataset = args.dataset

random.seed(0)

data = get_data(dataset)
g = to_networkx(data, to_undirected=True)

nn_emb = 6 * (len(g.nodes) // 10)
nn_train_lp = 3 * (len(g.nodes) // 10)
# Train emb - 60%, Train LP - 30%, Test LP - 10%

emb_nodes = list(random.sample(list(g.nodes), nn_emb))
train_nodes = list(random.sample(list(set(g.nodes) - set(emb_nodes)), nn_train_lp))
test_nodes = list(set(g.nodes) - set(emb_nodes) - set(train_nodes))

g_emb = g.subgraph(emb_nodes)

g_train = g.subgraph(train_nodes)

g_test = g.subgraph(test_nodes)

with open(f"./data/{args.dataset}/train_emb.pkl", "wb") as f:
    pickle.dump(emb_nodes, f)

with open(f"./data/{args.dataset}/train_nids.pkl", "wb") as f:
    pickle.dump(train_nodes, f)

with open(f"./data/{args.dataset}/test_nids.pkl", "wb") as f:
    pickle.dump(test_nodes, f)

edge_index = [[], []]
for u, v in g_emb.edges:
    edge_index[0].append(u)
    edge_index[1].append(v)

edge_index = torch.tensor(edge_index)

with open(f"./data/{args.dataset}/edge_index.pkl", "wb") as f:
    pickle.dump(edge_index, f)
