import json
import os.path
import zipfile

from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2
from keras.models import load_model
from util import release_gpu_memory, get_cv_data, get_custom_objects
from data_generator import SoccerNetTestVideoGenerator
from numpy import argmax, minimum, maximum, transpose, copy, load, array
from SoccerNet.Evaluation.ActionSpotting import evaluate
from numpy import max as np_max
from tqdm import tqdm


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


def test_soccernet(data, model_name: str = 'overall_best.hdf5', cv_iter: int = 0):
    os.makedirs(os.path.join("models", "SoccerNet", data["model"]), exist_ok=True)
    path = os.path.join("models", "SoccerNet", data["model"])

    if not os.path.exists(os.path.join(path, "checkpoints", f'{cv_iter}', model_name)):
        raise Exception(
            f"The model you tried to load does not exist: {os.path.join(path, 'checkpoints', f'{cv_iter}', model_name)}")

    model = load_model(os.path.join(path, "checkpoints", f'{cv_iter}', model_name),
                       custom_objects=get_custom_objects())

    games = get_cv_data("test", cv_iter)

    for game in tqdm(games):
        json_data = dict()
        json_data["UrlLocal"] = game
        json_data["predictions"] = list()
        for half in range(2):
            if "resnet" in data["model"].lower():
                train_generator = SoccerNetTestVideoGenerator(game, half + 1, data["batch size"], data["dataset path"],
                                                              data["feature fps"], data["window length"],
                                                              data["test stride"], data["frame dims"],
                                                              data["resize method"])
            elif "baidu" in data["model"].lower():
                train_generator = load(
                    os.path.join(data["dataset path"], game, f"{half + 1}_baidu_soccer_embeddings.npy"))
                temp = []
                num_chunks = (train_generator.shape[0] - (data["window length"] - data["test stride"])) // data[
                    "test stride"]
                for chunk in range(num_chunks):
                    temp.append(train_generator[
                                chunk * data["test stride"]: chunk * data["test stride"] + data["window length"], ...])
                train_generator = array(temp)

            pred_y = model.predict(x=train_generator, verbose=1, workers=data["workers"],
                                   use_multiprocessing=data["multi proc"], max_queue_size=data["queue size"])[:, 1:]

            for class_label in range(17):
                spots = get_spot_from_NMS(pred_y[:, class_label], 20 // data["test stride"])
                for spot in spots:
                    confidence = spot[1]

                    frame_index = spot[0]
                    total_seconds = frame_index * data["test stride"] + data["window length"] / 2
                    seconds = int(total_seconds % 60)
                    minutes = int(total_seconds // 60)

                    prediction_data = dict()
                    prediction_data["gameTime"] = str(half + 1) + " - " + str(minutes) + ":" + str(seconds)
                    prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[class_label]
                    prediction_data["position"] = str(int(total_seconds * 1000))
                    prediction_data["half"] = str(half + 1)
                    prediction_data["confidence"] = str(confidence)
                    json_data["predictions"].append(prediction_data)

            del train_generator
            release_gpu_memory()

        os.makedirs(os.path.join(path, "outputs_test", game), exist_ok=True)
        if os.path.exists(os.path.join(path, "outputs_test", game, "results_spotting.json")):
            os.remove(os.path.join(path, "outputs_test", game, "results_spotting.json"))
        with open(os.path.join(path, "outputs_test", game, "results_spotting.json"),
                  'w') as output_file:
            json.dump(json_data, output_file, indent=4)

    del model
    release_gpu_memory()

    # zip folder
    # zipResults(zip_path=os.path.join("models", "SoccerNet", data["model"], f"results_spotting_test.zip"),
    #            target_dir=os.path.join("models", "SoccerNet", data["model"], "outputs_test"),
    #            filename="results_spotting.json")

    results = evaluate(SoccerNet_path=data["dataset path"],
                       Predictions_path=os.path.join(path, "outputs_test"),
                       split="test",
                       prediction_file="results_spotting.json",
                       version=2)

    print(results)

    with open(os.path.join('models', "SoccerNet", data["model"], 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    return results["a_mAP"]
