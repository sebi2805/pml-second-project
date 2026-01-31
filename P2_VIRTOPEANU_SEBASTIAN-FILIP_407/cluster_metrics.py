# the overall strategy is too choose the most common label inside each predicted cluster


def majority_vote_mapping(labels_true, labels_pred):
    y_true = list(labels_true)
    y_pred = list(labels_pred)

    # count how mani times each true label shows up inside ech predicted cluster
    cluster_counts = {}
    min_len = min(len(y_true), len(y_pred))
    for i in range(min_len):
        true_label = y_true[i]
        pred_label = y_pred[i]

        if pred_label not in cluster_counts:
            cluster_counts[pred_label] = {}
        if true_label not in cluster_counts[pred_label]:
            cluster_counts[pred_label][true_label] = 0

        cluster_counts[pred_label][true_label] += 1

    mapping = {}
    for pred_label, counts in cluster_counts.items():
        top_count = 0
        for count in counts.values():
            if count > top_count:
                top_count = count

        # if there is a tie, pick the first one after sorting
        # (another idea was to pick based on the "recall" for example if there is a tie H vs M and in
        # this cluster we have 95% of values for H and 5% for M, then pick H since it is more likely to increase the overall accuracy)
        top_labels = []
        for label, count in counts.items():
            if count == top_count:
                top_labels.append(label)
        chosen = sorted(top_labels)[0]

        mapping[pred_label] = chosen
    return mapping


def majority_vote_counts(labels_true, labels_pred):
    y_true = list(labels_true)
    y_pred = list(labels_pred)

    cluster_counts = {}
    min_len = min(len(y_true), len(y_pred))
    for i in range(min_len):
        true_label = y_true[i]
        pred_label = y_pred[i]

        if pred_label not in cluster_counts:
            cluster_counts[pred_label] = {}
        if true_label not in cluster_counts[pred_label]:
            cluster_counts[pred_label][true_label] = 0

        cluster_counts[pred_label][true_label] += 1

    correct = 0
    for counts in cluster_counts.values():
        best = 0
        for count in counts.values():
            if count > best:
                best = count
        correct += best
    return correct, len(y_true)


def majority_vote_accuracy(labels_true, labels_pred):
    correct, total = majority_vote_counts(labels_true, labels_pred)
    return correct / total


def binary_match_counts(labels_true, labels_pred, positive_labels):
    y_true = list(labels_true)
    y_pred = list(labels_pred)

    cluster_counts = {}

    # for the binary case, create 2 bins [negative, positive]
    # sometimes we filter the noise in the dbscane, thats wth this additional step
    min_len = min(len(y_true), len(y_pred))
    for i in range(min_len):
        true_label = y_true[i]
        pred_label = y_pred[i]

        if pred_label not in cluster_counts:
            cluster_counts[pred_label] = [0, 0]
        counts = cluster_counts[pred_label]
        if true_label in positive_labels:
            counts[1] += 1
        else:
            counts[0] += 1

    correct = 0
    for counts in cluster_counts.values():
        if counts[1] > counts[0]:
            correct += counts[1]
        else:
            correct += counts[0]
    return correct, len(y_true)


def binary_accuracy(labels_true, labels_pred, positive_labels):
    correct, total = binary_match_counts(labels_true, labels_pred, positive_labels)
    return correct / total
