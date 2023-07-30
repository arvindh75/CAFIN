import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import pickle
import random
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import wandb
from experiments import run_exp
from model import *
from torch_cluster import random_walk
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_geometric.utils import to_networkx
from utils import *

parser = argparse.ArgumentParser(description="Train GraphSAGE for link prediction")
parser.add_argument("--dataset", type=str, help="The dataset", default="Cora")
parser.add_argument("--exp", type=int, help="Experiment number", default=0)
parser.add_argument(
    "--run", type=int, help="Run number to store results under", default="1000"
)
parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs to train unsupervised GraphSAGE",
    default=50,
)
parser.add_argument("--seed", type=int, help="Random seed value", default=0)
args = parser.parse_args()

NUM_CLASSES = num_classes(args.dataset)
PRINT = -1
WS = None
CLF = 1
NEG_MIN_DIST = 2
LR = 0.0025
BATCH_SIZE = 1024
NUM_LAYERS = 3
N_HIDDEN = 256
PASS = False
WANDB = False

if args.run == 1 and WANDB:
    wandb.init(
        project="Inductive Node Classification",
        entity="onlyfairness",
        config={
            "dataset": args.dataset,
            "exp": args.exp,
            "learning_rate": LR,
            "epochs": args.epochs,
            "n_hidden": N_HIDDEN,
            "num_layers": NUM_LAYERS,
            "seed": args.seed,
        },
    )
    wandb.run.name = (
        f"{args.dataset} {args.exp} {datetime.now().strftime('%d-%m %H:%M')}"
    )
    wandb.run.save()

torch.set_printoptions(profile="full")
torch.set_printoptions(precision=2)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
val = args.seed
random.seed(val)
np.random.seed(val)
torch.manual_seed(val)
torch.cuda.manual_seed(val)
gen = torch.Generator()
gen.manual_seed(val)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------


data = get_data(args.dataset)

with open(f"./data/{args.dataset}/edge_index.pkl", "rb") as f:
    edge_index = pickle.load(f)
data.edge_index = edge_index

G = to_networkx(data, to_undirected=True)

if PRINT > -1:
    print(nx.info(G))

NODES = len(G.nodes)
MAX_VAL = NODES - 1
degree = dict(G.degree())


if PRINT > -1:
    print(f"Importing distances")
dist = {}
with open(f"./data/{args.dataset}/dist.pkl", "rb") as f:
    dist = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x, edge_index = data.x.to(device), data.edge_index.to(device)


class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.as_tensor(batch)
        row, col, _ = self.adj_t.coo()
        rw = random_walk(row, col, batch, walk_length=1, coalesced=False)

        pos_batch = rw[:, 1]
        neg_batch = torch.randint(
            0, self.adj_t.size(1), (batch.numel(),), dtype=torch.long
        )

        for indx, u in enumerate(batch):
            u = int(u)

            v = int(pos_batch[indx])
            prev_v = v

            DEG.append(NODES / (degree[u] + 1))

            if PRINT >= 2:
                print(f"init pos -> u = {u}, v = {v}, dist = {dist[u][v]}")
            while dist[u][v] != 1:
                v = u
                # v = int(pos_batch[indx])
            if PRINT >= 2:
                if v != prev_v:
                    print(f"Changed")
                print(f"final pos -> u = {u}, v = {v}, dist = {dist[u][v]}")
            POS_DIST.append(dist[u][v] / MAX_VAL)

            v = int(neg_batch[indx])
            prev_v = v
            if PRINT >= 2:
                print(f"init neg -> u = {u}, v = {v}, dist = {dist[u][v]}")
            while dist[u][v] <= NEG_MIN_DIST:
                neg_batch[indx] = torch.randint(
                    0, self.adj_t.size(1), (), dtype=torch.long
                )
                v = int(neg_batch[indx])
            if PRINT >= 2:
                if v != prev_v:
                    print(f"Changed")
                print(f"final neg -> u = {u}, v = {v}, dist = {dist[u][v]}\n")
            NEG_DIST.append(dist[u][v] / MAX_VAL)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super().sample(batch)


def train(exp_no=0):
    global PASS
    if args.run == 1 and not PASS and WANDB:
        PASS = True
        wandb.watch(model, log_freq=1, log="all", log_graph=True)

    model.train()
    total_loss = 0

    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()

        out = model(x[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
        pos_vec_dist = torch.as_tensor(vec_euc_dist(out, pos_out), device=device)
        neg_vec_dist = torch.as_tensor(vec_euc_dist(out, neg_out), device=device)
        pos_loss, neg_loss = run_exp(
            exp_no,
            device,
            out,
            pos_out,
            neg_out,
            pos_vec_dist,
            neg_vec_dist,
            POS_DIST,
            NEG_DIST,
            DEG,
        )

        pos_loss = -pos_loss
        neg_loss = -neg_loss
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()
        POS_DIST.clear()
        NEG_DIST.clear()
        DEG.clear()
        total_loss += float(loss) * out.size(0)

        if args.run == 1 and WANDB:
            wandb.log(
                {
                    "pos_loss": pos_loss,
                    "neg_loss": neg_loss,
                    "loss": loss,
                }
            )

    scheduler.step()

    return total_loss / data.num_nodes


if __name__ == "__main__":
    print(f"\nEXP - {args.exp} RUN - {args.run}\n", flush=True)
    POS_DIST = []
    NEG_DIST = []
    DEG = []

    train_loader = NeighborSampler(
        data.edge_index,
        sizes=[15, 10],
        batch_size=BATCH_SIZE,
        generator=gen,
        shuffle=True,
        num_nodes=data.num_nodes,
    )
    model = SAGE(
        data.num_node_features, hidden_channels=N_HIDDEN, num_layers=NUM_LAYERS
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    for epoch in range(args.epochs):
        loss = train(args.exp)
        if PRINT >= 1:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
    if PRINT >= 1:
        print("--------------------------------------------------------")

    out = model.full_forward(x, edge_index).cpu()
    out = out.detach()
    with open(f"./data/{args.dataset}/embeds{args.exp}.pkl", "wb") as f:
        pickle.dump(out, f)
