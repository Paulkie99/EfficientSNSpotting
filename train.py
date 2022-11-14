import glob
import multiprocessing
import shutil
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
        verbose=1,
        save_best_only=True, save_weights_only=True)

    # Logging
    log_dir = join("models", "SoccerNet", data["model"], 'logs', f'{cv_iter}')
    makedirs(log_dir, exist_ok=True)
    csv_logger = CSVLogger(join(log_dir, 'training_' + iteration + '.log'), append=data["append training logs"])

    # Early stopping
    early_stopper = EarlyStopping(patience=data["patience"], verbose=1)

    # Tensorboard
    log_dir = join("models", "SoccerNet", data["model"], 'tensorboard', f'{cv_iter}')
    if exists(log_dir):
        shutil.rmtree(log_dir)
    makedirs(log_dir)
    tboard_callback = TensorBoard(log_dir=log_dir,
                                  profile_batch='2000,2100')
    tb_prog = program.TensorBoard()
    tb_prog.configure(argv=[None, '--logdir',
                            join(getcwd(), "models", "SoccerNet", data["model"], 'tensorboard', f'{cv_iter}')])
    tb_prog.launch()

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
            'batch_size': data["batch size"],
            'data_fraction': data["data fraction"]
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

    elif "baidu" in data["model"].lower() or "netvlad" in data["model"].lower():
        [remove(path) for path in glob.glob("*train_cache*")]
        [remove(path) for path in glob.glob("*valid_cache*")]

        train_generator = DeepFeatureGenerator(feature_type="baidu", window_len=data["window length"],
                                               stride=data["stride"], base_path=data["dataset path"],
                                               data_subset="train", cv_iter=cv_iter,
                                               fps=data["feature fps"],
                                               remove_replays=data["remove replays"],
                                               balance_classes=data["balance classes"],
                                               frame_dims=data["frame dims"],
                                               data_fraction=data["data fraction"])
        train_generator = tf.data.Dataset.from_generator(train_generator, output_signature=(
            tf.TensorSpec(shape=(data["window length"],
                                 data["frame dims"][1] - data["frame dims"][0]), dtype=tf.float32),
            tf.TensorSpec(shape=(18,), dtype=tf.uint8)
        ))
        train_generator = train_generator.take(-1).shuffle(2000).batch(data["batch size"],
                                                                       num_parallel_calls=tf.data.AUTOTUNE,
                                                                       deterministic=False,
                                                                       drop_remainder=(
                                                                           True if "netvlad" in
                                                                                   data[
                                                                                       "model"].lower() else False)).prefetch(
            tf.data.AUTOTUNE)

        validation_generator = DeepFeatureGenerator(feature_type="baidu", window_len=data["window length"],
                                                    stride=data["stride"], base_path=data["dataset path"],
                                                    data_subset="valid", cv_iter=cv_iter,
                                                    fps=data["feature fps"],
                                                    remove_replays=data["remove replays"],
                                                    balance_classes=data["balance classes"],
                                                    frame_dims=data["frame dims"],
                                                    data_fraction=data["data fraction"])
        validation_generator = tf.data.Dataset.from_generator(validation_generator, output_signature=(
            tf.TensorSpec(shape=(data["window length"],
                                 data["frame dims"][1] - data["frame dims"][0]), dtype=tf.float32),
            tf.TensorSpec(shape=(18,), dtype=tf.uint8)
        ))

        validation_generator = validation_generator.batch(data["batch size"],
                                                          num_parallel_calls=tf.data.AUTOTUNE,
                                                          deterministic=False,
                                                          drop_remainder=(True if "netvlad" in data[
                                                              "model"].lower() else False)).prefetch(
            tf.data.AUTOTUNE)

    history = model.fit(x=train_generator, verbose=1, callbacks=callbacks, epochs=data["epochs"],
                        use_multiprocessing=data["multi proc"],
                        workers=data["workers"], validation_data=validation_generator,
                        initial_epoch=data["init epoch"], validation_freq=1, max_queue_size=data["queue size"])

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

    queue = multiprocessing.Queue()

    for cv_iter in tqdm(range(data["CV start"], cv_iterations)):
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

        for i in range(data["MC start"], data["MC iterations"]):
            print(f"Starting MC iteration {i + 1}/{data['MC iterations']}, of CV iteration "
                  f"{cv_iter + 1}/{cv_iterations}")

            p = multiprocessing.Process(target=train, args=(data, i, cv_iter, queue))
            p.start()
            history = queue.get()
            p.join()
            print("Joined")

            save_metrics["epochs"].append(len(history['val_loss']))
            index_of_min_validation_loss = argmin(history['val_loss'])
            for k in history.keys():
                save_metrics[k].append(history[k][index_of_min_validation_loss])

            test_accuracy = test_soccernet(data, f'best_{i}.hdf5', cv_iter=cv_iter)
            save_metrics["test a-mAP"].append(test_accuracy)

            if min(history['val_loss']) < save_metrics["best_val_loss"]:
                save_metrics["best_val_loss"] = min(history['val_loss'])
                save_metrics["best_val_iter"] = i
                rename(join('models', "SoccerNet", data["model"], "checkpoints", f'{cv_iter}', f'best_{i}.hdf5'),
                       join('models', "SoccerNet", data["model"], "checkpoints", f'{cv_iter}', f'overall_best.hdf5'))
            else:
                remove(join('models', "SoccerNet", data["model"], "checkpoints", f'{cv_iter}', f'best_{i}.hdf5'))

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
    cv_avg_metrics["best cv iter"] = int(argmin(cv_avg_metrics["best_val_loss"]))
    cv_avg_metrics["best iter"] = cv_avg_metrics["best_val_iter"][cv_avg_metrics["best cv iter"]]
    cv_avg_metrics["best valid loss"] = cv_avg_metrics["best_val_loss"][cv_avg_metrics["best cv iter"]]

    for k in range(cv_iterations):
        if k != cv_avg_metrics["best cv iter"]:
            remove(join('models', "SoccerNet", data["model"], "checkpoints", f'{k}', 'overall_best.hdf5'))

    with open(join('models', "SoccerNet", data["model"], "results", "CV", f'avg_metrics_all_cv.json'), 'w') as f:
        json.dump(cv_avg_metrics, f, indent=4)


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
