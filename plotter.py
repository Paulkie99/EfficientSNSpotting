import json
import os
from os.path import join

import matplotlib.pyplot as plt
from numpy import where, array


def plot_train_history(data):
    with open(join('models', "SoccerNet", data["model"], "results", "CV", f'avg_metrics_all_cv.json'), 'r') as f:
        metrics = json.load(f)

    log_dir = join("models", "SoccerNet", data["model"], 'logs', f'{metrics["best cv iter"]}',
                   f'training_{metrics["best iter"]}.log')

    with open(log_dir) as f:
        data_rows = f.read().split('\n')
        if data_rows[-1] == '':
            data_rows.pop(-1)
        f.close()

    indices = []
    metrics = data["train metrics"] + ['loss']
    for k in metrics:
        indices.append(where(array(data_rows[0].split(',')) == k)[0][0])
        indices.append(where(array(data_rows[0].split(',')) == 'val_' + k)[0][0])
    data_rows.pop(0)

    lists = [list() for i in range(len(indices))]
    for row in data_rows:
        split_row = row.split(',')
        for index in range(len(indices)):
            lists[index].append(float(split_row[indices[index]]))

    plt.style.use('ggplot')

    save_path = os.path.join("models", "SoccerNet", data["model"], "results", "figures")
    os.makedirs(save_path, exist_ok=True)

    for i in range(0, len(indices), 2):
        plt.plot(lists[i], marker='.', markersize=4, linestyle='--')
        plt.plot(lists[i + 1], marker='.', markersize=4, linestyle='--')
        plt.title(f"model {metrics[i // 2]}")
        plt.ylabel(metrics[i // 2])
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig(os.path.join(save_path, f"{metrics[i // 2]}.png"), bbox_inches='tight',
                    dpi=300)
        plt.close()
