import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Planetoid, Twitch


def get_data(dataset_name):
    path = f"./data/{dataset_name}"
    if dataset_name == "Cora":
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name == "CiteSeer":
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name == "EN":
        dataset = Twitch(path, "EN", transform=T.NormalizeFeatures())
    elif dataset_name == "Photo":
        dataset = Amazon(path, "Photo", transform=T.NormalizeFeatures())
    elif dataset_name == "Computers":
        dataset = Amazon(path, "Computers", transßform=T.NormalizeFeatures())
    else:
        print(
            'Invalid dataset name, must be in ["Cora", "CiteSeer", "EN", "Photo", "Computers"]'
        )
        exit()

    return dataset[0]


def num_classes(dataset_name):
    if dataset_name == "Cora":
        return 7
    elif dataset_name == "CiteSeer":
        return 6
    elif dataset_name == "EN":
        return 2
    elif dataset_name == "Photo":
        return 8
    elif dataset_name == "Computers":
        return 10


def vec_euc_dist(x, y):
    return (x - y).pow(2).sum(-1).sqrt()


def compute_distance(u: int, v: int, dc: dict):
    if u == v:
        return 1
    INF = 10**5
    if dc["comp_id"][u] != dc["comp_id"][v]:
        return INF
    elif (u, v) in dc["edges"]:
        return 1
    else:
        mn = INF
        for a, b in zip(dc["distances"][u], dc["distances"][v]):
            mn = min(mn, a + b)
        return mn
