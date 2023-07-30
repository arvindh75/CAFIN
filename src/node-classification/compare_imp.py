import argparse
import statistics

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="The dataset", default="Cora")
args = parser.parse_args()

exps = [0, 17, 18, 19]
data = [[] for _ in range(len(exps))]
imparity_deg = [[] for _ in range(len(exps))]
imparity_deg_sd = [[] for _ in range(len(exps))]
top_imparity_deg = [0 for _ in range(len(exps))]
accuracies_deg = [0 for _ in range(len(exps))]
imparity_eig = [[] for _ in range(len(exps))]
imparity_eig_sd = [[] for _ in range(len(exps))]
top_imparity_eig = [0 for _ in range(len(exps))]
accuracies_eig = [0 for _ in range(len(exps))]
NUM_CLASSES = num_classes(args.dataset)

TOP_K = [0.2, 0.5, 1]

for indx, exp in enumerate(exps):
    # Deg
    with open(f"res/{args.dataset}/EXP{exp}/imparity_deg.txt", "r") as f:
        temp = sorted([float(line) for line in f.readlines()])
        data[indx] = temp
    for k in TOP_K:
        imparity_deg[indx].append(
            sum(data[indx][: max(1, int(k * len(data[indx])))])
            / max(1, int(k * len(data[indx])))
        )
        imparity_deg_sd[indx].append(
            statistics.pstdev(data[indx][: max(1, int(k * len(data[indx])))])
        )
        top_imparity_deg[indx] = data[indx][0]

    # Eig
    with open(f"res/{args.dataset}/EXP{exp}/imparity_eig.txt", "r") as f:
        temp = sorted([float(line) for line in f.readlines()])
        data[indx] = temp
    for k in TOP_K:
        imparity_eig[indx].append(
            sum(data[indx][: max(1, int(k * len(data[indx])))])
            / max(1, int(k * len(data[indx])))
        )
        imparity_eig_sd[indx].append(
            statistics.pstdev(data[indx][: max(1, int(k * len(data[indx])))])
        )
        top_imparity_eig[indx] = data[indx][0]

    with open(f"res/{args.dataset}/EXP{exp}/acc_eig.txt", "r") as f:
        temp = [float(line) for line in f.readlines()]
        accuracies_eig[indx] = sum(temp) / len(temp)
    with open(f"res/{args.dataset}/EXP{exp}/acc_deg.txt", "r") as f:
        temp = [float(line) for line in f.readlines()]
        accuracies_deg[indx] = sum(temp) / len(temp)


for indx, exp in enumerate(exps):
    print("Degree Centrality")
    for i, k in enumerate(TOP_K):
        print(
            f"Improvement for exp {exp} averaged over the top {k*100}% of runs: {(1 - (imparity_deg[indx][i]/imparity_deg[0][i]))*100}% \
              \nCoefficient of Variance: {100*imparity_deg_sd[indx][i]/imparity_deg[indx][i]}%"
        )
    print(
        f"Improvement for exp {exp} for the best run:  {(1 - (top_imparity_deg[indx]/top_imparity_deg[0]))*100}%"
    )
    print(f"Accuracy for exp {exp}: {accuracies_deg[indx]*100}%")
    print(
        f"Accuracy change for exp {exp}: {(accuracies_deg[indx] - accuracies_deg[0])*100}%"
    )
    print()

    print("Eigen Centrality")
    for i, k in enumerate(TOP_K):
        print(
            f"Improvement for exp {exp} averaged over the top {k*100}% of runs: {(1 - (imparity_eig[indx][i]/imparity_eig[0][i]))*100}% \
              \nCoefficient of Variance: {100*imparity_eig_sd[indx][i]/imparity_eig[indx][i]}%"
        )
    print(
        f"Improvement for exp {exp} for the best run:  {(1 - (top_imparity_eig[indx]/top_imparity_eig[0]))*100}%"
    )

    print(f"Accuracy for exp {exp}: {accuracies_eig[indx]*100}%")
    print(
        f"Accuracy change for exp {exp}: {(accuracies_eig[indx] - accuracies_eig[0])*100}%"
    )
    print()
