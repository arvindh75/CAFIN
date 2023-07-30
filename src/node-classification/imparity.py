from itertools import combinations


def check_bucket(nid, c_dict, splits):
    val = c_dict[str(nid)]
    if val <= splits[0]:
        return 0
    else:
        return 1


def imparity_new(class_acc_final, NUM_CLASSES):
    combs = list(combinations([0, 1], 2))
    total = []
    for i in range(NUM_CLASSES):
        for u, v in combs:
            if class_acc_final[i][u] == -1 or class_acc_final[i][v] == -1:
                continue
            total.append(abs(class_acc_final[i][u] - class_acc_final[i][v]))
    if len(total):
        return sum(total) / len(total)
    else:
        return -1


def create_weights(y_test):
    weights = {}
    for i in y_test:
        i = int(i)
        if i in weights:
            weights[i] += 1
        else:
            weights[i] = 1
    return weights


def imparity_weighted(class_acc_final, weights, NUM_CLASSES):
    combs = list(combinations([0, 1], 2))
    total = 0
    total_weight = 0
    went = 0
    for i in range(NUM_CLASSES):
        for u, v in combs:
            if class_acc_final[i][u] == -1 or class_acc_final[i][v] == -1:  #
                continue
            went = 1
            total += abs(class_acc_final[i][u] - class_acc_final[i][v]) * weights[i]
            total_weight += weights[i]

    if total_weight:
        return total / total_weight
    else:
        return -1


def inter_class_acc(
    test_nids, y_pred_test, y_test, c_dict, splits, NUM_CLASSES, printt=False
):
    class_acc = [[[], []] for _ in range(NUM_CLASSES)]
    class_acc_final = [[0, 0] for _ in range(NUM_CLASSES)]

    for indx, nid in enumerate(test_nids):
        b = check_bucket(nid, c_dict, splits)
        class_acc[y_test[indx]][b].append(
            float(int(y_pred_test[indx]) == int(y_test[indx]))
        )

    for i in range(NUM_CLASSES):
        for j in range(2):
            if len(class_acc[i][j]):
                class_acc_final[i][j] = sum(class_acc[i][j]) / len(class_acc[i][j])
            else:
                class_acc_final[i][j] = -1

    if printt:
        print()
        for i in range(2):
            print(f"Bucket {i}:")
            for j in range(NUM_CLASSES):
                print(f"class {j} - {class_acc_final[j][i]}")
            print()
        print()

    return class_acc_final


def overall_acc(
    test_nids, y_pred_test, y_test, c_dict, splits, NUM_CLASSES, printt=False
):
    class_acc = [[[], []] for _ in range(NUM_CLASSES)]

    for indx, nid in enumerate(test_nids):
        b = check_bucket(nid, c_dict, splits)
        class_acc[y_test[indx]][b].append(
            float(int(y_pred_test[indx]) == int(y_test[indx]))
        )

    correct = 0
    total = 0

    for i in range(NUM_CLASSES):
        for j in range(2):
            if len(class_acc[i][j]):
                correct += sum(class_acc[i][j])
                total += len(class_acc[i][j])

    return correct / total
