import json
import os
from os.path import join

import matplotlib.pyplot as plt
from numpy import where, array


def plot_train_history(arch: str = 'ResNet', dataset: str = 'SoccerNet'):
    with open(join('models', dataset, arch, "results", "CV", f'avg_metrics_all_cv.json'), 'r') as f:
        metrics = json.load(f)

    log_dir = join("models", dataset, arch, 'logs', f'{metrics["best cv iter"]}', f'training_{metrics["best iter"]}.log')

    with open(log_dir) as f:
        data_rows = f.read().split('\n')
        if data_rows[-1] == '':
            data_rows.pop(-1)
        f.close()

    accuracy_idx = where(array(data_rows[0].split(',')) == 'accuracy')[0][0]
    loss_idx = where(array(data_rows[0].split(',')) == 'loss')[0][0]
    val_accuracy_idx = where(array(data_rows[0].split(',')) == 'val_accuracy')[0][0]
    val_loss_idx = where(array(data_rows[0].split(',')) == 'val_loss')[0][0]
    data_rows.pop(0)

    accuracy = []
    val_accuracy = []
    loss = []
    val_loss = []
    for row in data_rows:
        split_row = row.split(',')
        accuracy.append(float(split_row[accuracy_idx]))
        val_accuracy.append(float(split_row[val_accuracy_idx]))
        loss.append(float(split_row[loss_idx]))
        val_loss.append(float(split_row[val_loss_idx]))

    plt.style.use('ggplot')

    save_path = os.path.join("models", dataset, arch, "results", "figures")
    os.makedirs(save_path, exist_ok=True)
    plt.plot(accuracy, marker='.', markersize=4, linestyle='--')
    plt.plot(val_accuracy, marker='.', markersize=4, linestyle='--')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(os.path.join(save_path, "accuracy.png"), bbox_inches='tight', dpi=300)
    plt.close()

    plt.plot(loss, marker='.', markersize=4, linestyle='--')
    plt.plot(val_loss, marker='.', markersize=4, linestyle='--')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(save_path, "loss.png"), bbox_inches='tight', dpi=300)
    plt.close()
