import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pickle
import json
import argparse
from torch_geometric.utils import to_networkx
from sklearn.svm import LinearSVC
import pandas as pd
from imparity import *
from utils import *

parser = argparse.ArgumentParser(description='Train GraphSAGE for node classification')
parser.add_argument('--dataset', type=str, help='Dataset name', default='Cora')
parser.add_argument('--exp', type=int, help='Experiment number', default='0')
parser.add_argument('--seed', type=int, help='Random seed value', default=0)
args = parser.parse_args()


data = get_data(args.dataset)
G = to_networkx(data, to_undirected=True)
NUM_CLASSES = num_classes(args.dataset)

degc_dict = {}
with open(f"./data/{args.dataset}/deg_c.json") as f:
    degc_dict = json.load(f)

eigc_dict = {}
with open(f"./data/{args.dataset}/eig_c.json") as f:
    eigc_dict = json.load(f)

dist = {}
with open(f"./data/{args.dataset}/dist.pkl", "rb") as f:
    dist = pickle.load(f)

degree = dict(G.degree())

df1 = pd.DataFrame.from_dict(list(degc_dict.items()))
df1 = df1.rename(columns= {0 : "nid", 1: "deg_c"})
df2 = pd.DataFrame.from_dict(list(eigc_dict.items()))
df2 = df2.rename(columns= {0 : "nid", 1: "eig_c"})

df4 = pd.merge(df1, df2, on="nid")

deg_q = list(df4["deg_c"].quantile([0.25, 0.5, 0.75]))
eig_q = list(df4["eig_c"].quantile([0.25, 0.5, 0.75]))

deg_split = [deg_q[1]]
eig_split = [eig_q[1]]

with open(f"./data/{args.dataset}/embeds{args.exp}.pkl", "rb") as f:
    out = pickle.load(f)

with open(f"./data/{args.dataset}/train_nids.pkl", "rb") as f:
    train_nids = pickle.load(f)

with open(f"./data/{args.dataset}/test_nids.pkl", "rb") as f:
    test_nids = pickle.load(f)


train_mask = [False for i in range(len(G.nodes))]
test_mask = [False for i in range(len(G.nodes))]
for node in train_nids:
    train_mask[int(node)] = True
for node in test_nids:
    test_mask[int(node)] = True    

clf = LinearSVC(random_state=args.seed)
clf.fit(out[train_mask], data.y[train_mask])

    
y_pred_test = clf.predict(out[test_mask])
test_nids, y_test = [], []
to_train = []

for indx, i in enumerate(test_mask):
    if i:
        to_train.append({
            "x" : out[indx].detach().cpu().tolist(),
            "y": int(data.y[indx])
        })
        test_nids.append(indx)
        y_test.append(data.y[indx])

run_class_acc_eig = inter_class_acc(test_nids, y_pred_test, y_test, eigc_dict, eig_split, NUM_CLASSES, False)
overall_acc_eig = overall_acc(test_nids, y_pred_test, y_test, eigc_dict, eig_split, NUM_CLASSES, False)
run_class_acc_deg = inter_class_acc(test_nids, y_pred_test, y_test, degc_dict, deg_split, NUM_CLASSES, False)
overall_acc_deg = overall_acc(test_nids, y_pred_test, y_test, degc_dict, deg_split, NUM_CLASSES, False)

WS = create_weights(y_test)

eig_imp = imparity_weighted(run_class_acc_eig, WS, NUM_CLASSES)
deg_imp = imparity_weighted(run_class_acc_deg, WS, NUM_CLASSES)

with open(f"./res/{args.dataset}/EXP{args.exp}/imparity_eig.txt", "a") as f:
    print(f"{eig_imp}", file=f)
with open(f"./res/{args.dataset}/EXP{args.exp}/imparity_deg.txt", "a") as f:
    print(f"{deg_imp}", file=f)

with open(f"./res/{args.dataset}/EXP{args.exp}/class_acc_eig.txt", "a") as f:
    print(f"{run_class_acc_eig}", file=f)
with open(f"./res/{args.dataset}/EXP{args.exp}/class_acc_deg.txt", "a") as f:
    print(f"{run_class_acc_deg}", file=f)

with open(f"./res/{args.dataset}/EXP{args.exp}/acc_eig.txt", "a") as f:
    print(f"{overall_acc_eig}", file=f)
with open(f"./res/{args.dataset}/EXP{args.exp}/acc_deg.txt", "a") as f:
    print(f"{overall_acc_deg}", file=f)