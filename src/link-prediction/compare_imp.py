import argparse
import pickle
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="The dataset", default="Cora")
args = parser.parse_args()

with open(f"./data/{args.dataset}/test_labels.pkl", "rb") as f:
    y_test = pickle.load(f)
bucket_sizes = []
for i in range(3):
    bucket_sizes.append(len(y_test[i][0]) + len(y_test[i][1]))

exps = [0, 17, 18, 19]
data = [[] for _ in range(len(exps))]
imparity = [[] for _ in range(len(exps))]
imparity_sd = [[] for _ in range(len(exps))]
top_imparity = [0 for _ in range(len(exps))]
accuracies = [0 for _ in range(len(exps))]


TOP_K = [0.2, 0.5, 1]

for indx, exp in enumerate(exps):
    with open(f"res/{args.dataset}/EXP{exp}/imparity.txt", "r") as f:
        temp = sorted([float(line) for line in f.readlines()])
        data[indx] = temp
    for k in TOP_K:
        imparity[indx].append(
            sum(data[indx][: max(1, int(k * len(data[indx])))])
            / max(1, int(k * len(data[indx])))
        )
        imparity_sd[indx].append(
            statistics.pstdev(data[indx][: max(1, int(k * len(data[indx])))])
        )
        top_imparity[indx] = data[indx][0]

    acc = 0
    for i in range(3):
        with open(f"res/{args.dataset}/EXP{exp}/cls_acc/{i}.txt", "r") as f:
            temp = [float(line) for line in f.readlines()]
            acc += (sum(temp) / len(temp)) * bucket_sizes[i]
    acc /= sum(bucket_sizes)
    accuracies[indx] = acc

for indx, exp in enumerate(exps):
    for i, k in enumerate(TOP_K):
        print(
            f"Improvement for exp {exp} averaged over the top {k*100}% of runs: {(1 - (imparity[indx][i]/imparity[0][i]))*100}% \
              \nCoefficient of Variance: {100*imparity_sd[indx][i]/imparity[indx][i]}%"
        )
    print(
        f"Improvement for exp {exp} for the best run:  {(1 - (top_imparity[indx]/top_imparity[0]))*100}%"
    )
    print(f"Accuracy for exp {exp}: {accuracies[indx]*100}%")
    print(f"Accuracy change for exp {exp}: {(accuracies[indx] - accuracies[0])*100}%")
    print()
