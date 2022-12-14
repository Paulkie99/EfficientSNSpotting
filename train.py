import glob
import multiprocessing
import shutil
import time

from tensorboard import program
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, History, TensorBoard
from os.path import join, exists
from os import getcwd, remove, makedirs, rename
import tensorflow as tf
from tqdm import tqdm
from data_generator import SoccerNetTrainVideoDataGenerator, DeepFeatureGenerator, \
    SoccerNetTrainDataset
from util import release_gpu_memory, save_train_latex_table, save_test_latex_table, setup_environment, \
    map_train_metrics_to_funcs, get_config
from models import create_model
from numpy import inf, argmin, mean, std, sqrt, square
from test import test_soccernet
from plotter import plot_train_history
import json


def train(data, iteration, cv_iter, queue) -> History:
    """

    :return: Training history.
    """
    model = create_model(data)
    if data["saved model"] != "":
        model.load_weights(data["saved model"])
    optimizer = Adam(learning_rate=float(data["learning rate"]), decay=float(data["decay"]))
    model.compile(optimizer,
                  loss='binary_crossentropy',
                  metrics=map_train_metrics_to_funcs(data["train metrics"]))

    iteration = str(iteration)

    """
    Define checkpoints such that the model will be saved after each epoch. Save both best model and latest so that
    training may continue but we always have access to the best model.
    """
    checkpoints_dir = join("models", "SoccerNet", data["model"], "checkpoints", f'{cv_iter}')
    makedirs(checkpoints_dir, exist_ok=True)
    best_checkpointer = ModelCheckpoint(
        filepath=join(checkpoints_dir, 'best_' + iteration + '.hdf5'),
        verbose=(1 if iteration == "0" else 0),
        save_best_only=True, save_weights_only=True)

    # Logging
    log_dir = join("models", "SoccerNet", data["model"], 'logs', f'{cv_iter}')
    makedirs(log_dir, exist_ok=True)
    csv_logger = CSVLogger(join(log_dir, 'training_' + iteration + '.log'), append=data["append training logs"])

    # Early stopping
    early_stopper = EarlyStopping(patience=data["patience"], verbose=1)

    callbacks = [best_checkpointer, csv_logger, early_stopper]

    if iteration == "0":
        model.summary()
        # Tensorboard
        log_dir = join("models", "SoccerNet", data["model"], 'tensorboard', f'{cv_iter}')
        if exists(log_dir):
            shutil.rmtree(log_dir)
        makedirs(log_dir)
        tboard_callback = TensorBoard(log_dir=log_dir,
                                      profile_batch='100,300')
        tb_prog = program.TensorBoard()
        tb_prog.configure(argv=[None, '--logdir',
                                join(getcwd(), "models", "SoccerNet", data["model"], 'tensorboard', f'{cv_iter}')])
        tb_prog.launch()
        callbacks.append(tboard_callback)

    [remove(path) for path in glob.glob("*train_cache*")]
    [remove(path) for path in glob.glob("*valid_cache*")]

    train_generator = DeepFeatureGenerator(data, cv_iter=cv_iter, data_subset="train")
    validation_generator = DeepFeatureGenerator(data, cv_iter=cv_iter, data_subset="valid")
    if 'frames' in data[
        "features"].lower():  # All images cannot fit into memory, a generator must be used for training.
        train_generator = tf.data.Dataset.from_generator(train_generator, output_signature=(
            tf.TensorSpec(shape=(data["window length"],
                                 data["frame dims"][1], data["frame dims"][0], 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(18,), dtype=tf.uint8)
        ))

        train_generator = train_generator.map(lambda x, y: (tf.divide(x, 255), y), num_parallel_calls=tf.data.AUTOTUNE)
        if data["resize method"] != "":
            # TODO check desired resize method and apply
            pass

        validation_generator = tf.data.Dataset.from_generator(validation_generator, output_signature=(
            tf.TensorSpec(shape=(data["window length"],
                                 data["frame dims"][1], data["frame dims"][0], 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(18,), dtype=tf.uint8)
        ))

        validation_generator = validation_generator.map(lambda x, y: (tf.divide(x, 255), y),
                                                        num_parallel_calls=tf.data.AUTOTUNE)
        if data["resize method"] != "":
            # TODO check desired resize method and apply
            pass

    elif "baidu" in data["features"].lower():
        train_generator = tf.data.Dataset.from_generator(train_generator, output_signature=(
            tf.TensorSpec(shape=(data["window length"],
                                 data["frame dims"][1] - data["frame dims"][0]), dtype=tf.float32),
            tf.TensorSpec(shape=(18,), dtype=tf.uint8)
        ))

        validation_generator = tf.data.Dataset.from_generator(validation_generator, output_signature=(
            tf.TensorSpec(shape=(data["window length"],
                                 data["frame dims"][1] - data["frame dims"][0]), dtype=tf.float32),
            tf.TensorSpec(shape=(18,), dtype=tf.uint8)
        ))

    train_generator = train_generator.take(-1).shuffle(2 * data["batch size"]).batch(data["batch size"],
                                                                                     num_parallel_calls=tf.data.AUTOTUNE,
                                                                                     deterministic=False).prefetch(
        tf.data.AUTOTUNE)
    validation_generator = validation_generator.batch(data["batch size"],
                                                      num_parallel_calls=tf.data.AUTOTUNE,
                                                      deterministic=False).prefetch(tf.data.AUTOTUNE)

    history = model.fit(x=train_generator, verbose=(1 if iteration == "0" else 0), callbacks=callbacks,
                        epochs=data["epochs"],
                        validation_data=validation_generator,
                        initial_epoch=data["init epoch"], validation_freq=1)

    [remove(path) for path in glob.glob("*train_cache*")]
    [remove(path) for path in glob.glob("*valid_cache*")]

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

    all_cv_start = time.time()
    for cv_iter in tqdm(range(data["CV start"], cv_iterations)):
        cv_start = time.time()
        save_metrics = {}
        for k in data["train metrics"]:
            save_metrics[k] = []
            save_metrics['val_' + k] = []
        save_metrics["loss"] = []
        save_metrics["val_loss"] = []
        save_metrics["epochs"] = []
        save_metrics["best_val_loss"] = inf
        save_metrics["best_val_iter"] = 0
        save_metrics["test a-mAP"] = []

        for i in range(data["MC start"], data["MC iterations"], data["workers"]):

            stop_idx = min(i + data["workers"], data["MC iterations"])
            start = time.time()
            print(f"Starting MC iteration {i + 1}--{stop_idx}/{data['MC iterations']}, of CV iteration "
                  f"{cv_iter + 1}/{cv_iterations}")

            # Start training
            processes = [object for worker in range(i, stop_idx)]
            queues = [multiprocessing.Queue() for worker in range(i, stop_idx)]
            histories = [0] * (stop_idx - i)
            for j in range(i, stop_idx):
                processes[j - i] = multiprocessing.Process(target=train, args=(data, j, cv_iter, queues[j - i]))
                processes[j - i].start()

            # Wait for training to finish
            for j in range(i, stop_idx):
                histories[j - i] = queues[j - i].get()
                processes[j - i].join()
                print(f"Process {j} joined")

            # Start testing
            test_queues = [multiprocessing.Queue() for worker in range(i, stop_idx)]
            for j in range(i, min(i + data["workers"], data["MC iterations"])):
                processes[j - i] = multiprocessing.Process(target=test_soccernet, args=(data, f'best_{j}.hdf5', cv_iter,
                                                                                        test_queues[j - i]))
                processes[j - i].start()

            # Record metrics when testing is finished
            for j in range(i, stop_idx):
                test_accuracy = test_queues[j - i].get()
                processes[j - i].join()
                history = histories[j - i]

                save_metrics["epochs"].append(len(history['val_loss']))
                index_of_min_validation_loss = argmin(history['val_loss'])
                for k in history.keys():
                    save_metrics[k].append(history[k][index_of_min_validation_loss])

                save_metrics["test a-mAP"].append(test_accuracy)

                if min(history['val_loss']) < save_metrics["best_val_loss"]:
                    save_metrics["best_val_loss"] = min(history['val_loss'])
                    save_metrics["best_val_iter"] = j
                    if exists(join('models', "SoccerNet", data["model"], "checkpoints", f'{cv_iter}',
                                   f'overall_best.hdf5')):
                        remove(join('models', "SoccerNet", data["model"], "checkpoints", f'{cv_iter}',
                                    f'overall_best.hdf5'))
                    rename(join('models', "SoccerNet", data["model"], "checkpoints", f'{cv_iter}', f'best_{j}.hdf5'),
                           join('models', "SoccerNet", data["model"], "checkpoints", f'{cv_iter}',
                                f'overall_best.hdf5'))
                else:
                    remove(join('models', "SoccerNet", data["model"], "checkpoints", f'{cv_iter}', f'best_{j}.hdf5'))

            stop = time.time() - start
            hours = stop // 3600
            minutes = (stop - (hours * 3600)) // 60
            seconds = (stop - (hours * 3600)) % 60
            print(f"Ended MC iterations {i + 1}--{stop_idx}, time taken: {hours:.2f} hours, "
                  f"{minutes:.2f} minutes and {seconds:.2f} seconds.")

        stop = time.time() - cv_start
        hours = (stop // 3600)
        minutes = (stop - (hours * 3600)) // 60
        seconds = (stop - (hours * 3600)) % 60
        print(f"Ended CV iteration {cv_iter + 1}, time taken: {hours:.2f} hours, "
              f"{minutes:.2f} minutes and {seconds:.2f} seconds.")

        to_save = {}
        for k in save_metrics.keys():
            if "best" not in k:
                to_save[k] = mean(save_metrics[k])
                to_save[k + ' std'] = std(save_metrics[k])
            else:
                to_save[k] = save_metrics[k]

        makedirs(join('models', "SoccerNet", data["model"], "results", "CV"), exist_ok=True)
        with open(join('models', "SoccerNet", data["model"], "results", "CV", f'avg_metrics_cv{cv_iter}.json'),
                  'w') as f:
            json.dump(to_save, f, indent=4)

    stop = time.time() - all_cv_start
    hours = (stop // 3600)
    minutes = (stop - (hours * 3600)) // 60
    seconds = (stop - (hours * 3600)) % 60
    print(f"Ended all CV iterations, time taken: {hours:.2f} hours, "
          f"{minutes:.2f} minutes and {seconds:.2f} seconds.")

    cv_avg_metrics = {}
    for cv_iter in range(cv_iterations):
        with open(join('models', "SoccerNet", data["model"], "results", "CV", f'avg_metrics_cv{cv_iter}.json'),
                  'r') as f:
            metrics = json.load(f)
            for k in metrics.keys():
                if cv_avg_metrics.get(k, None) is None:
                    cv_avg_metrics[k] = []
                cv_avg_metrics[k].append(metrics[k])

    for k in cv_avg_metrics.keys():
        # if "std" in k:
        #     if data["MC iterations"] > 1:
        #         cv_avg_metrics[k] = sqrt(sum(square(cv_avg_metrics[k])) / cv_iterations)
        #     else:
        #         cv_avg_metrics[k] = std(cv_avg_metrics[k])
        if "best" not in k and "std" not in k:
            temp = cv_avg_metrics[k]
            cv_avg_metrics[k] = mean(temp)
            cv_avg_metrics[k + ' std'] = std(temp)
    cv_avg_metrics["best cv iter"] = int(argmin(cv_avg_metrics["best_val_loss"]))
    cv_avg_metrics["best iter"] = cv_avg_metrics["best_val_iter"][cv_avg_metrics["best cv iter"]]
    cv_avg_metrics["best valid loss"] = cv_avg_metrics["best_val_loss"][cv_avg_metrics["best cv iter"]]

    with open(join('models', "SoccerNet", data["model"], "results", "CV", f'avg_metrics_all_cv.json'), 'w') as f:
        json.dump(cv_avg_metrics, f, indent=4)

    for k in range(cv_iterations):
        if k != cv_avg_metrics["best cv iter"]:
            remove(join('models', "SoccerNet", data["model"], "checkpoints", f'{k}', 'overall_best.hdf5'))


if __name__ == '__main__':
    with open("config.json", "r") as jsonfile:
        data = json.load(jsonfile)

    data = get_config(data)

    setup_environment(data)

    if not data["test only"]:
        train_for_iterations(data)

    # Test results, plotting and tables

    with open(join('models', "SoccerNet", data["model"], "results", "CV", f'avg_metrics_all_cv.json'), 'r') as f:
        metrics = json.load(f)

    if data["CV iterations"] or data["MC iterations"] > 1 or data["test only"]:
        test_soccernet(data, cv_iter=metrics["best cv iter"])

    plot_train_history(data)
    save_train_latex_table(data)
    save_test_latex_table(data)
