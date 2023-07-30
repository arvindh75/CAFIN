import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import json
import pickle
import random

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_networkx
from utils import *

random.seed(69)

parser = argparse.ArgumentParser(description="Train GraphSAGE for link prediction")
parser.add_argument("--dataset", type=str, help="The dataset", default="Cora")
args = parser.parse_args()
data = get_data(args.dataset)

G = to_networkx(data, to_undirected=True)
NODES = len(G.nodes)

degc_dict = {}
with open(f"./data/{args.dataset}/deg_c.json") as f:
    degc_dict = json.load(f)

eigc_dict = {}
with open(f"./data/{args.dataset}/eig_c.json") as f:
    eigc_dict = json.load(f)

degree = dict(G.degree())

df1 = pd.DataFrame.from_dict(list(degc_dict.items()))
df1 = df1.rename(columns={0: "nid", 1: "deg_c"})
df2 = pd.DataFrame.from_dict(list(eigc_dict.items()))
df2 = df2.rename(columns={0: "nid", 1: "eig_c"})

df4 = pd.merge(df1, df2, on="nid")

deg_q = list(df4["deg_c"].quantile([0.25, 0.5, 0.75]))
eig_q = list(df4["eig_c"].quantile([0.25, 0.5, 0.75]))

deg_split = [deg_q[1]]
eig_split = [eig_q[1]]

indx_g1_deg = [i for i in range(NODES) if degc_dict[str(i)] <= deg_split[0]]
indx_g2_deg = [i for i in range(NODES) if degc_dict[str(i)] > deg_split[0]]

indx_g1_eig = [i for i in range(NODES) if eigc_dict[str(i)] <= eig_split[0]]
indx_g2_eig = [i for i in range(NODES) if eigc_dict[str(i)] > eig_split[0]]


buckets_deg = [0, 0, 0]
edge_list_deg = [[], [], []]
for edge in G.edges:
    c = 0
    if edge[0] in indx_g2_deg:
        c += 1
    if edge[1] in indx_g2_deg:
        c += 1
    buckets_deg[c] += 1
    edge_list_deg[c].append(edge)


train_emb = []
train_lp = [[[], []] for i in range(3)]
test_lp = [[[], []] for i in range(3)]


for pop in range(len(edge_list_deg)):
    te, lp, _, _ = train_test_split(
        edge_list_deg[pop], [0] * buckets_deg[pop], test_size=0.4, random_state=42
    )
    train_emb.extend(te)
    tlp, telp, _, _ = train_test_split(
        lp, [0] * len(lp), test_size=0.25, random_state=42
    )
    train_lp[pop][0].extend(tlp)
    test_lp[pop][0].extend(telp)

total_size = len(G.edges)

picked = set()
for pop in range(3):
    while len(train_lp[pop][0]) != len(train_lp[pop][1]):
        sampled = []
        temp = pop
        while len(sampled) < 2:
            if temp:
                element = random.sample(indx_g2_deg, 1)[0]
                temp -= 1
            else:
                element = random.sample(indx_g1_deg, 1)[0]
            sampled.append(element)
        used = False
        for i in range(2):
            if (
                (sampled[i], sampled[1 - i]) in picked
                or (sampled[i], sampled[1 - i]) in G.edges
                or sampled[i] == sampled[1 - i]
            ):
                used = True
        if used:
            continue
        picked.add((sampled[0], sampled[1]))
        train_lp[pop][1].append((sampled[0], sampled[1]))

for pop in range(3):
    while len(test_lp[pop][0]) != len(test_lp[pop][1]):
        sampled = []
        temp = pop
        while len(sampled) < 2:
            if temp:
                element = random.sample(indx_g2_deg, 1)[0]
                temp -= 1
            else:
                element = random.sample(indx_g1_deg, 1)[0]
            sampled.append(element)
        used = False
        for i in range(2):
            if (
                (sampled[i], sampled[1 - i]) in picked
                or (sampled[i], sampled[1 - i]) in G.edges
                or sampled[i] == sampled[1 - i]
            ):
                used = True
        if used:
            continue
        picked.add((sampled[0], sampled[1]))
        test_lp[pop][1].append((sampled[0], sampled[1]))

train_x_lp = []
train_y_lp = []
test_x_lp = test_lp
test_y_lp = []

for i in range(3):
    train_x_lp.extend(train_lp[i][0])
    train_y_lp.extend([1] * len(train_lp[i][0]))
    train_x_lp.extend(train_lp[i][1])
    train_y_lp.extend([0] * len(train_lp[i][1]))

    test_y_lp.append([])
    test_y_lp[-1].append([1] * len(test_lp[i][0]))
    test_y_lp[-1].append([0] * len(test_lp[i][1]))


with open(f"./data/{args.dataset}/train_emb.pkl", "wb") as f:
    pickle.dump(train_emb, f)

with open(f"./data/{args.dataset}/train_nids.pkl", "wb") as f:
    pickle.dump(train_x_lp, f)

with open(f"./data/{args.dataset}/train_labels.pkl", "wb") as f:
    pickle.dump(train_y_lp, f)

with open(f"./data/{args.dataset}/test_nids.pkl", "wb") as f:
    pickle.dump(test_x_lp, f)

with open(f"./data/{args.dataset}/test_labels.pkl", "wb") as f:
    pickle.dump(test_y_lp, f)


with open(f"./data/{args.dataset}/train_emb.pkl", "rb") as f:
    edges = pickle.load(f)

edge_index = [[], []]
for u, v in edges:
    edge_index[0].append(u)
    edge_index[1].append(v)

edge_index = torch.tensor(edge_index)

with open(f"./data/{args.dataset}/edge_index.pkl", "wb") as f:
    pickle.dump(edge_index, f)
