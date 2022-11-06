import multiprocessing
import os
import shutil

from keras_nlp.layers import TransformerEncoder
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, History, TensorBoard
from keras.models import load_model, save_model
from tensorflow.keras.metrics import AUC
from os.path import join
import tensorflow as tf
from tqdm import tqdm
from data_generator import SoccerNetTrainVideoDataGenerator, TransformerTrainFeatureGenerator, \
    SoccerNetTrainDataset
from util import release_gpu_memory, save_train_latex_table, save_test_latex_table, create_model, PositionalEmbedding, \
    setup_environment
from numpy import inf, argmin, mean, std, sqrt, square
from test import test_soccernet
from plotter import plot_train_history
import json


def train(data, iteration, cv_iter, queue) -> History:
    """

    :return: Training history.
    """
    model = (
        create_model(data) if data["saved model"] == "" else load_model(data["saved model"], custom_objects={
            'PositionalEmbedding': PositionalEmbedding,
            'TransformerEncoder': TransformerEncoder}))
    optimizer = Adam(learning_rate=float(data["learning rate"]), decay=float(data["decay"]))
    model.compile(optimizer,
                  loss='binary_crossentropy',
                  metrics=[AUC(multi_label=True), 'accuracy'])

    iteration = str(iteration)

    """
    Define checkpoints such that the model will be saved after each epoch. Save both best model and latest so that
    training may continue but we always have access to the best model.
    """
    checkpoints_dir = os.path.join("models", "SoccerNet", data["model"], "checkpoints", f'{cv_iter}')
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_checkpointer = ModelCheckpoint(
        filepath=join(checkpoints_dir, 'best_' + iteration + '.hdf5'),
        verbose=1,
        save_best_only=True)

    # Logging
    log_dir = join("models", "SoccerNet", data["model"], 'logs', f'{cv_iter}')
    os.makedirs(log_dir, exist_ok=True)
    csv_logger = CSVLogger(join(log_dir, 'training_' + iteration + '.log'), append=data["append training logs"])

    # Early stopping
    early_stopper = EarlyStopping(patience=data["patience"], verbose=1)

    # Tensorboard
    log_dir = join("models", "SoccerNet", data["model"], 'tensorboard', f'{cv_iter}')
    tboard_callback = TensorBoard(log_dir=log_dir)
    # profile_batch='10,129')

    callbacks = [best_checkpointer, csv_logger, early_stopper, tboard_callback]

    if 'resnet' in data["model"].lower():  # All images cannot fit into memory, a generator must be used for training.
        params = {
            'train': 'train',
            'data_path': data["dataset path"],
            'fps': data["feature fps"],
            'window_len': data["window length"],
            'cv_iter': cv_iter,
            'frame_dims': data["frame dims"],
            'resize_method': data["resize method"],
            'batch_size': data["batch size"]
        }

        train_generator = SoccerNetTrainVideoDataGenerator(**params)
        # gen = SoccerNetTrainDataset(**params)
        # train_generator = tf.data.Dataset.from_generator(gen, output_signature=(
        #     tf.TensorSpec(shape=(int(ceil(window_len * feature_fps)), 224, 398, 3), dtype=tf.uint8),
        #     tf.TensorSpec(shape=(18,), dtype=tf.uint8)
        # ))
        # train_generator = train_generator.map((lambda x, y: (tf.divide(x, 255), y)),
        #                                       num_parallel_calls=tf.data.AUTOTUNE)
        # train_generator = train_generator.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE,
        #                                         deterministic=False).prefetch(tf.data.AUTOTUNE)

        params['train'] = 'valid'
        validation_generator = SoccerNetTrainVideoDataGenerator(**params)
        # gen = SoccerNetTrainDataset(**params)
        # validation_generator = tf.data.Dataset.from_generator(gen, output_signature=(
        #     tf.TensorSpec(shape=(int(ceil(window_len * feature_fps)), 224, 398, 3), dtype=tf.uint8),
        #     tf.TensorSpec(shape=(18,), dtype=tf.uint8)
        # ))
        # validation_generator = validation_generator.map((lambda x, y: (tf.divide(x, 255), y)),
        #                                                 num_parallel_calls=tf.data.AUTOTUNE)
        # validation_generator = validation_generator.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE,
        #                                                   deterministic=False).prefetch(tf.data.AUTOTUNE)

    elif data["model"] == "Baidu":
        train_generator = TransformerTrainFeatureGenerator(feature_type="baidu", window_len=data["window length"],
                                                           stride=data["stride"], base_path=data["dataset path"],
                                                           data_subset="train", cv_iter=cv_iter)
        train_generator = tf.data.Dataset.from_generator(train_generator, output_signature=(
            tf.TensorSpec(shape=(data["window length"], 8576), dtype=tf.uint8),
            tf.TensorSpec(shape=(18,), dtype=tf.uint8)
        ))
        train_generator = train_generator.batch(data["batch size"], num_parallel_calls=tf.data.AUTOTUNE,
                                                deterministic=False).prefetch(tf.data.AUTOTUNE)

        validation_generator = TransformerTrainFeatureGenerator(feature_type="baidu", window_len=data["window length"],
                                                                stride=data["stride"], base_path=data["dataset path"],
                                                                data_subset="valid", cv_iter=cv_iter)
        validation_generator = tf.data.Dataset.from_generator(validation_generator, output_signature=(
            tf.TensorSpec(shape=(data["window length"], 8576), dtype=tf.uint8),
            tf.TensorSpec(shape=(18,), dtype=tf.uint8)
        ))
        validation_generator = validation_generator.batch(data["batch size"], num_parallel_calls=tf.data.AUTOTUNE,
                                                          deterministic=False).prefetch(tf.data.AUTOTUNE)

    history = model.fit(x=train_generator, verbose=1, callbacks=callbacks, epochs=data["epochs"],
                        use_multiprocessing=data["multi proc"],
                        workers=data["workers"], validation_data=validation_generator,
                        initial_epoch=data["init epoch"], validation_freq=1, max_queue_size=data["queue size"])

    del train_generator
    del validation_generator
    del model
    release_gpu_memory()

    queue.put(history.history)


def train_for_iterations(data):
    if data["CV iterations"]:
        cv_iterations = 5
    else:
        cv_iterations = 1

    queue = multiprocessing.Queue()

    for cv_iter in tqdm(range(data["CV start"], cv_iterations)):
        best_val_loss = inf
        best_val_iter = 0
        train_accuracies = []
        train_loss = []
        test_accuracies = []
        val_accuracies = []
        val_loss = []
        train_auc = []
        val_auc = []
        epochs = []

        for i in range(data["MC start"], data["MC iterations"]):
            print(f"Starting MC iteration {i + 1}/{data['MC iterations']}, of CV iteration "
                  f"{cv_iter + 1}/{cv_iterations}")

            p = multiprocessing.Process(target=train, args=(data, i, cv_iter, queue))
            p.start()
            history = queue.get()
            p.join()
            print("Joined")

            epochs.append(len(history['val_loss']))
            index_of_min_validation_loss = argmin(history['val_loss'])
            train_accuracies.append(history['accuracy'][index_of_min_validation_loss])
            train_loss.append(history['loss'][index_of_min_validation_loss])
            val_loss.append(history['val_loss'][index_of_min_validation_loss])
            val_accuracies.append(history['val_accuracy'][index_of_min_validation_loss])
            train_auc.append(history['auc'][index_of_min_validation_loss])
            val_auc.append(history['val_auc'][index_of_min_validation_loss])

            if min(history['val_loss']) < best_val_loss:
                best_val_loss = min(history['val_loss'])
                best_val_iter = i
                load_path = join('models', "SoccerNet", data["model"], "checkpoints", f'{cv_iter}', f'best_{i}.hdf5')
                save_path = join('models', "SoccerNet", data["model"], "checkpoints", f'{cv_iter}', f'overall_best.hdf5')
                shutil.copy(load_path, save_path)

            test_accuracy = test_soccernet(data, f'best_{i}.hdf5', cv_iter=cv_iter)
            test_accuracies.append(test_accuracy)

        to_save = {
            "train acc": mean(train_accuracies),
            "train acc std": std(train_accuracies),
            "train loss": mean(train_loss),
            "train loss std": std(train_loss),
            "train auc": mean(train_auc),
            "train auc std": std(train_auc),
            "test acc": mean(test_accuracies),
            "test acc std": std(test_accuracies),
            "valid acc": mean(val_accuracies),
            "valid acc std": std(val_accuracies),
            "valid loss": mean(val_loss),
            "valid loss std": std(val_loss),
            "valid auc": mean(val_auc),
            "valid auc std": std(val_auc),
            "epochs": mean(epochs),
            "epochs std": std(epochs),
            "best valid loss": best_val_loss,
            "best iter": best_val_iter
        }
        os.makedirs(join('models', "SoccerNet", data["model"], "results", "CV"), exist_ok=True)
        with open(join('models', "SoccerNet", data["model"], "results", "CV", f'avg_metrics_cv{cv_iter}.json'),
                  'w') as f:
            json.dump(to_save, f, indent=4)

    cv_avg_metrics = {}
    with open(join('models', "SoccerNet", data["model"], "results", "CV", f'avg_metrics_cv{0}.json'), 'r') as f:
        metrics = json.load(f)
        for k in metrics.keys():
            cv_avg_metrics[k] = []
    for cv_iter in range(cv_iterations):
        with open(join('models', "SoccerNet", data["model"], "results", "CV", f'avg_metrics_cv{cv_iter}.json'),
                  'r') as f:
            metrics = json.load(f)
            for k in metrics.keys():
                cv_avg_metrics[k].append(metrics[k])

    for k in cv_avg_metrics.keys():
        if "std" in k:
            if data["MC iterations"] > 1:
                cv_avg_metrics[k] = sqrt(sum(square(cv_avg_metrics[k])) / cv_iterations)
            else:
                cv_avg_metrics[k] = std(cv_avg_metrics[k])
        elif "best" not in k:
            cv_avg_metrics[k] = mean(cv_avg_metrics[k])
    cv_avg_metrics["best cv iter"] = int(argmin(cv_avg_metrics["best valid loss"]))
    cv_avg_metrics["best iter"] = cv_avg_metrics["best iter"][cv_avg_metrics["best cv iter"]]
    cv_avg_metrics["best valid loss"] = cv_avg_metrics["best valid loss"][cv_avg_metrics["best cv iter"]]

    with open(join('models', "SoccerNet", data["model"], "results", "CV", f'avg_metrics_all_cv.json'), 'w') as f:
        json.dump(cv_avg_metrics, f, indent=4)


if __name__ == '__main__':
    with open("config.json", "r") as jsonfile:
        data = json.load(jsonfile)

    setup_environment(data)

    train_for_iterations(data)

    # Test results, plotting and tables

    with open(join('models', "SoccerNet", data["model"], "results", "CV", f'avg_metrics_all_cv.json'), 'r') as f:
        metrics = json.load(f)

    test_soccernet(data, cv_iter=metrics["best cv iter"])

    plot_train_history(arch=data["model"], dataset="SoccerNet")
    save_train_latex_table(dataset="SoccerNet")
    save_test_latex_table(dataset="SoccerNet")
