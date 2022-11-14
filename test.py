import json
import os.path
import shutil
import zipfile

from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2
from util import release_gpu_memory, get_cv_data
from models import create_model
from data_generator import SoccerNetTestVideoGenerator, DeepFeatureGenerator
from numpy import argmax, minimum, maximum, transpose, copy
from SoccerNet.Evaluation.ActionSpotting import evaluate
from numpy import max as np_max
import tensorflow as tf


def zipResults(zip_path, target_dir, filename="results_spotting.json"):
    zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            if file == filename:
                fn = os.path.join(base, file)
                zipobj.write(fn, fn[rootlen:])


def get_spot_from_NMS(class_predictions_whole_vid, window=30, thresh=0.5):
    detections_tmp = copy(class_predictions_whole_vid)
    indices = []
    max_vals = []
    while max(detections_tmp) >= thresh:
        # Get the max remaining index and value
        max_value = np_max(detections_tmp)
        max_index = argmax(detections_tmp)
        max_vals.append(max_value)
        indices.append(max_index)
        # detections_NMS[max_index,i] = max_value

        nms_from = int(maximum(-(window / 2) + max_index, 0))
        nms_to = int(minimum(max_index + int(window / 2), len(detections_tmp)))
        detections_tmp[nms_from:nms_to] = -1

    return transpose([indices, max_vals])


def test_soccernet(data, model_name: str = 'overall_best.hdf5', cv_iter: int = 0, queue=None):
    os.makedirs(os.path.join("models", "SoccerNet", data["model"]), exist_ok=True)
    path = os.path.join("models", "SoccerNet", data["model"])

    if model_name == "overall_best.hdf5":
        iteration = -1
    else:
        iteration = int(model_name.split('.')[0].split('_')[1])

    if not os.path.exists(os.path.join(path, "checkpoints", f'{cv_iter}', model_name)):
        raise Exception(
            f"The model you tried to load does not exist: {os.path.join(path, 'checkpoints', f'{cv_iter}', model_name)}")

    model = create_model(data)
    model.load_weights(os.path.join(path, "checkpoints", f'{cv_iter}', model_name))

    games = get_cv_data("test", cv_iter, data["data fraction"])
    if "video" in data["features"].lower():
        # TODO change test generator
        pass
        # train_generator = SoccerNetTestVideoGenerator(game[1], half + 1, data["batch size"], data["dataset path"],
        #                                               data["feature fps"], data["window length"],
        #                                               data["test stride"], data["frame dims"],
        #                                               data["resize method"])

    elif "baidu" in data["features"].lower():
        generator = DeepFeatureGenerator(data, cv_iter=cv_iter, data_subset="test")
        train_generator = tf.data.Dataset.from_generator(generator, output_signature=(
            tf.TensorSpec(shape=(None, data["window length"], data["frame dims"][1] - data["frame dims"][0]),
                          dtype=tf.float32)
        ))
        train_generator = train_generator.prefetch(tf.data.AUTOTUNE)

        for vid in enumerate(games):
            assert os.path.join(data["dataset path"], vid[1]) == generator.feature_paths[2 * vid[0]][0]

    all_pred_y = []
    print("Testing")
    for x in train_generator:
        all_pred_y.append(model.predict(x=x, batch_size=data["batch size"], verbose=0)[:, 1:])

    assert len(all_pred_y) == 200

    del model
    del train_generator
    release_gpu_memory()

    for game in enumerate(games):
        json_data = dict()
        json_data["UrlLocal"] = game[1]
        json_data["predictions"] = list()
        for half in range(2):
            pred_y = all_pred_y[2 * game[0] + half]

            for class_label in range(17):
                spots = get_spot_from_NMS(pred_y[:, class_label],
                                          (data["NMS window"] * data["feature fps"]) // data["test stride"],
                                          data["NMS threshold"])
                for spot in spots:
                    confidence = spot[1]

                    frame_index = spot[0]
                    total_seconds = (frame_index * data["test stride"] + data["window length"] / 2) / data[
                        "feature fps"]
                    seconds = int(total_seconds % 60)
                    minutes = int(total_seconds // 60)

                    prediction_data = dict()
                    prediction_data["gameTime"] = str(half + 1) + " - " + str(minutes) + ":" + str(seconds)
                    prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[class_label]
                    prediction_data["position"] = str(int(total_seconds * 1000))
                    prediction_data["half"] = str(half + 1)
                    prediction_data["confidence"] = str(confidence)
                    json_data["predictions"].append(prediction_data)

        os.makedirs(os.path.join(path, "outputs_test", f'{iteration}', game[1]), exist_ok=True)
        if os.path.exists(os.path.join(path, "outputs_test", f'{iteration}', game[1], "results_spotting.json")):
            os.remove(os.path.join(path, "outputs_test", f'{iteration}', game[1], "results_spotting.json"))
        with open(os.path.join(path, "outputs_test", f'{iteration}', game[1], "results_spotting.json"),
                  'w') as output_file:
            json.dump(json_data, output_file, indent=4)

    results = evaluate(SoccerNet_path=data["dataset path"],
                       Predictions_path=os.path.join(path, "outputs_test", f'{iteration}'),
                       split="test",
                       prediction_file="results_spotting.json",
                       version=2, cv_iter=cv_iter, data_fraction=data["data fraction"],
                       framerate=data["feature fps"])

    if iteration == -1:
        with open(os.path.join(path, "outputs_test", "best_results.json"), 'w') as f:
            json.dump(results, f, indent=4)
    else:
        shutil.rmtree(os.path.join(path, "outputs_test", f'{iteration}'))

    print(results)

    if queue is not None:
        queue.put(results["a_mAP"])
    else:
        return results["a_mAP"]


if __name__ == "__main__":
    data = {
        "model": "Netvlad++ b-features test 4",
        "model params": {},
        "saved model": "",
        "init epoch": 0,

        "test only": False,
        "MC iterations": 5,
        "MC start": 0,
        "CV iterations": True,
        "CV start": 0,
        "learning rate": 1e-4,
        "decay": 0,
        "batch size": 256,
        "epochs": 30,
        "patience": 5,
        "train metrics": ["auc", "precision", "recall", "accuracy"],
        "remove replays": False,
        "balance classes": False,
        "dataset path": "F:\\SoccerNet",
        "data fraction": 0.5,

        "feature fps": 1.0,
        "frame dims": [0, 8576],
        "resize method": "",
        "window length": 7,
        "stride": 7,
        "test stride": 1,

        "NMS window": 30,
        "NMS threshold": 0,

        "jit": True,
        "append training logs": False,
        "workers": 3,

        "config check exclusions": ["saved model", "init epoch", "test only", "MC start", "CV start", "train metrics",
                                    "dataset path", "test stride", "jit", "append training logs", "multi proc",
                                    "workers", "queue size"]
    }
    test_soccernet(data)
