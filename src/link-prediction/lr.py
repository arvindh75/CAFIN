import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="Train GraphSAGE for link prediction")
parser.add_argument("--dataset", type=str, help="Dataset name", default="Cora")
parser.add_argument("--exp", type=int, help="Experiment number", default="0")
parser.add_argument("--seed", type=int, help="Random seed value", default=69)
args = parser.parse_args()


with open(f"./data/{args.dataset}/embeds{args.exp}.pkl", "rb") as f:
    embeds = pickle.load(f)


def f_join(u, v, typ=0):
    if typ == 0:
        return torch.cat((u, v), -1).numpy()
    else:
        return torch.mean(torch.stack([u, v]))


with open(f"./data/{args.dataset}/train_nids.pkl", "rb") as f:
    train_nids = pickle.load(f)

with open(f"./data/{args.dataset}/test_nids.pkl", "rb") as f:
    test_nids = pickle.load(f)

x_train = []
for nid in train_nids:
    x_train.append(f_join(embeds[nid[0]], embeds[nid[1]], 0))

x_test = []
for bucket in test_nids:
    x_test.append([])
    x_test[-1].append([])
    for nid in bucket[0]:
        x_test[-1][-1].append(f_join(embeds[nid[0]], embeds[nid[1]], 0))
    x_test[-1].append([])
    for nid in bucket[1]:
        x_test[-1][-1].append(f_join(embeds[nid[0]], embeds[nid[1]], 0))

with open(f"./data/{args.dataset}/train_labels.pkl", "rb") as f:
    y_train = pickle.load(f)

with open(f"./data/{args.dataset}/test_labels.pkl", "rb") as f:
    y_test = pickle.load(f)


val = 42
clf = LogisticRegression(random_state=args.seed)
clf.fit(x_train, y_train)

scores = []
for i in range(3):
    y_pred = np.append(clf.predict(x_test[i][0]), clf.predict(x_test[i][1]))
    y_orig = np.append(y_test[i][0], y_test[i][1])
    scores.append(
        (clf.score(x_test[i][0], y_test[i][0]) + clf.score(x_test[i][1], y_test[i][1]))
        * 0.5
    )
    with open(f"res/{args.dataset}/EXP{args.exp}/cls_acc/{i}.txt", "a") as f:
        print(
            (
                clf.score(x_test[i][0], y_test[i][0])
                + clf.score(x_test[i][1], y_test[i][1])
            )
            * 0.5,
            file=f,
        )
    with open(f"res/{args.dataset}/EXP{args.exp}/con_mat/{i}.txt", "a") as f:
        print(confusion_matrix(y_pred, y_orig), file=f)

mean = sum(scores) / len(scores)
sd = (sum([((x - mean) ** 2) for x in scores]) / len(scores)) ** 0.5
with open(f"res/{args.dataset}/EXP{args.exp}/imparity.txt", "a") as f:
    print(sd, file=f)
