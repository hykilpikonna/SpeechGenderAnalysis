import json

if __name__ == '__main__':

    with open(r'C:\Datasets\CN-Celeb_flac\id_labels.json', 'r', encoding='UTF-8') as f:
        labels = json.load(f)

    with open(r'C:\Datasets\CN-Celeb_flac\ina_pf_map.json', 'r', encoding='UTF-8') as f:
        pf = json.load(f)

    print(len(labels))

    correct_f = []
    correct_m = []
    incorrect_f = []
    incorrect_m = []

    for k in labels:
        if k not in pf:
            print(f'Skipped {k}')
            continue

        if labels[k] == 'f':
            if pf[k] > 0.5:
                correct_f.append(k)
            else:
                incorrect_f.append(k)

        if labels[k] == 'm':
            if pf[k] < 0.5:
                correct_m.append(k)
            else:
                incorrect_m.append(k)

    print('Done Reading\n')

    tp = len(correct_f)
    tn = len(correct_m)
    fp = len(incorrect_f)
    fn = len(incorrect_m)

    print('True Positive (F classified as F):', tp)
    print('True Negative (M classified as M):', tn)
    print('False Positive (F classified as M):', fp)
    print('False Negative (M classified as F):', fn)

    acc = (tp + tn) / (tp + tn + fp + fn)
    precision_f = tp / (tp + fp)
    recall_f = tp / (tp + fn)
    precision_m = tn / (tn + fn)
    recall_m = tn / (tn + fp)

    print('Accuracy:', acc)
    print('Precision F:', precision_f)
    print('Recall F:', recall_f)
    print('Precision M:', precision_m)
    print('Recall M:', recall_m)

    print('F wrongly classified as M:', fp / (tp + fp))
    print('M wrongly classified as F:', fn / (tn + fn))



