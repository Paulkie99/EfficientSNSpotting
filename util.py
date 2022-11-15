import gc
import json
import os
import shutil
from itertools import cycle, islice
from math import ceil
import cv2
import h5py
from keras.metrics import AUC, Precision, Recall
from numpy import save, zeros, single, array, where, load
from os.path import join, exists
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D
from tqdm import tqdm
from keras.backend import clear_session
from tabulate import tabulate
from SoccerNet.Downloader import getListGames, SoccerNetDownloader
from skvideo.measure import scenedet
from skvideo.io import vread
from bisect import bisect_left
import sys
sys.path.append("./VGGish-master")

from vggish import VGGish
from preprocess_sound import preprocess_sound
from moviepy.editor import VideoFileClip
from tensorflow.config.experimental import list_physical_devices, set_memory_growth
from tensorflow.keras import mixed_precision
from os import environ
from tensorflow.config.optimizer import set_jit


def get_config(data):
    path = "all_configs.json"
    if exists(path):
        with open(path, 'r') as f:
            config_list = json.load(f)
    else:
        config_list = {"configs": []}
    for config in range(len(config_list["configs"])):
        if isConfigEqual(data, config_list["configs"][config], data["config check exclusions"]):
            data["model"] = config_list["configs"][config]["model"]
            return data
    data["model"] = data["model"] + ' ' + str(len(config_list["configs"]))
    config_list["configs"].append(data)
    with open(path, 'w') as f:
        json.dump(config_list, f, indent=4)
    return data


def isConfigEqual(conf1, conf2, exclusions):
    for k, v in conf1.items():
        if k == "model":
            if v.split(' ') != conf2[k].split(' ')[:-1]:
                return False
        elif k not in exclusions:
            if v != conf2[k]:
                return False
    return True


def setup_environment(data):
    # tf.config.run_functions_eagerly(True)
    environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
    environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    gpus = list_physical_devices('GPU')
    for gpu in gpus:
        set_memory_growth(gpu, True)
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    if data["jit"]:
        set_jit(True)
    release_gpu_memory()


def getDuration(video_path):
    """Get the duration (in frames) for a video.

    Keyword arguments:
    video_path -- the path of the video
    """
    with VideoFileClip(video_path) as clip:
        return clip.reader.nframes


def getDurationSeconds(video_path):
    """Get the duration (in seconds) for a video.

    Keyword arguments:
    video_path -- the path of the video
    """
    cap = cv2.VideoCapture(video_path)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    duration = total / fps
    return float(duration)


def delete_soccernet_frames():
    for vid in tqdm(getListGames(["train", "test", "valid", "challenge"])):
        for half in range(1, 3):
            if exists(join("E:\\SoccerNet", vid, f"{half}_frames.h5")):
                os.remove(join("E:\\SoccerNet", vid, f"{half}_frames.h5"))
            if exists(join("E:\\SoccerNet", vid, f"{half}_scene_starts.npy")):
                os.remove(join("E:\\SoccerNet", vid, f"{half}_scene_starts.npy"))
            if exists(join("E:\\SoccerNet", vid, f"{half}_resnet_predictions.npy")):
                os.remove(join("E:\\SoccerNet", vid, f"{half}_resnet_predictions.npy"))


def check_extract_soccernet_frames():
    downloader = SoccerNetDownloader("E:\\SoccerNet")
    downloader.password = input("SoccerNet API password: ")
    for vid in tqdm(getListGames(["train", "test", "valid"])):
        for half in range(1, 3):
            if not exists(join("E:\\SoccerNet", vid, f"{half}_224p.mkv")):
                downloader.downloadGames(files=[f"{half}_224p.mkv"],
                                         split=["train", "test", "valid"])
            if exists(join("H:\\SoccerNet", vid, f"{half}_frames.h5")):
                try:
                    with VideoFileClip(join("E:\\SoccerNet", vid, f"{half}_224p.mkv"), audio=False) as clip:
                        duration = clip.reader.nframes
                    with h5py.File(join("H:\\SoccerNet", vid, f"{half}_frames.h5"), 'r') as hf:
                        if abs(ceil(duration / 8) - hf['soccernet_3_125_fps'].shape[0]) <= 1:
                            continue
                        else:
                            print(f"\n{join(vid, f'{half}_frames.h5')} is NOT fine.")
                except Exception as e:
                    print(f"\n{e}")

            with VideoFileClip(join("E:\\SoccerNet", vid, f"{half}_224p.mkv"), audio_fps=False) as cap:
                frames = []
                for i in range(ceil(cap.reader.nframes / 8)):
                    frames.append(cap.reader.read_frame())
                    cap.reader.skip_frames(7)
                frames = array(frames)

                assert abs(frames.shape[0] - ceil(cap.reader.nframes / 8)) <= 1

            os.makedirs(join("H:\\SoccerNet", vid), exist_ok=True)
            with h5py.File(join("H:\\SoccerNet", vid, f"{half}_frames.h5"), 'w') as h5f:
                h5f.create_dataset('soccernet_3_125_fps', data=frames)


def get_cv_data(subset, cv_iter, data_fraction):
    """
    Read appropriate data split.
    """
    paths = getListGames(["train"])
    paths.extend(getListGames(["valid"]))
    paths.extend(getListGames(["test"]))
    assert len(paths) == 500
    paths = cycle(paths)
    ret_paths = []
    # folds = 5
    # thus, each fold is 100 samples in size
    shift = 100 * cv_iter
    if subset == "train":  # training set is 300 samples
        ret_paths = list(islice(paths, shift, shift + 300))
    elif subset == "valid":  # valid set is offset by 300 samples
        shift += 300
        ret_paths = list(islice(paths, shift, shift + 100))
    elif subset == "test":  # test set is offset by 400 samples
        shift += 400
        ret_paths = list(islice(paths, shift, shift + 100))

    return ret_paths[:int(data_fraction * len(ret_paths))]


def crop_frame(ret):
    # ret = imutils.resize(ret, height=224)  # keep aspect ratio
    # number of pixel to remove per side
    if len(ret.shape) == 3:  # single image
        off_side = int((ret.shape[1] - 224) / 2)
        ret = ret[:, off_side:-off_side, :]  # remove them
    elif len(ret.shape) == 4:  # one sample (16 frames)
        off_side = int((ret.shape[2] - 224) / 2)
        ret = ret[:, :, off_side:-off_side, :]  # remove them
    elif len(ret.shape) == 5:  # batch
        off_side = int((ret.shape[3] - 224) / 2)
        ret = ret[:, :, :, off_side:-off_side, :]  # remove them
    return ret


def resize(frames, method, frame_dims):
    if method != "":
        if method.lower() == "crop":
            return crop_frame(frames)
        if method == 'INTER_AREA':
            inter = cv2.INTER_AREA
        for i in range(frames.shape[0]):
            frames[i, ...] = cv2.resize(frames[i, ...], (frame_dims[1], frame_dims[0]),
                                        interpolation=inter)
    return frames


def save_train_latex_table(data_):
    base_len = len(join("models", "SoccerNet")) + 1
    headers = ['Model']
    metrics = data_["train metrics"] + ['loss']

    data = []
    for i in metrics:
        data.append(["Training " + i[0].upper() + i[1:]])
        data.append(["Validation " + i[0].upper() + i[1:]])
    data += [["Test a-mAP"]] + [["Epochs"]]
    for base, dirs, files in os.walk(join("models", "SoccerNet")):
        for file in files:
            if file == "avg_metrics_all_cv.json":
                path = join(base, file)
                with open(path, 'r') as f:
                    jdata = json.load(f)
                    headers.append(f'{path[base_len:-len(file) - len("results") - len("CV") - 3]}')
                    for i in range(len(metrics)):
                        data[2 * i].append(f"{jdata[metrics[i]]:.4f} \u00B1 {jdata[metrics[i] + ' std']:.4f}")
                        data[2 * i + 1].append(
                            f"{jdata[f'val_{metrics[i]}']:.4f} \u00B1 {jdata[f'val_{metrics[i]} std']:.4f}")

                    data[-2].append(f"{jdata['test a-mAP']:.4f} \u00B1 {jdata['test a-mAP std']:.4f}")
                    data[-1].append(f"{jdata['epochs']:.4f} \u00B1 {jdata['epochs std']:.4f}")

    with open(join("models", "SoccerNet", f'all_models_train_metrics_latex.txt'), 'w') as f:
        f.write(tabulate(data, headers=headers, tablefmt='latex', floatfmt=".4f"))


def save_test_latex_table(data_):
    base_len = len(join("models", "SoccerNet")) + 1
    headers = ['Model/Metric'] + [f'Class {i}' for i in range(17)]
    data = []
    for base, dirs, files in os.walk(join("models", "SoccerNet")):
        for file in files:
            if file == "best_results.json":
                path = join(base, file)
                with open(path, 'r') as f:
                    jdata = json.load(f)
                    for k, v in jdata.items():
                        add_data = [f'{path[base_len:-len(file) - 1]}/{k}']
                        if isinstance(v, list):
                            add_data.extend(v)
                        else:
                            add_data.append(v)
                        if len(data) == 0:
                            data = [add_data]
                        else:
                            data.append(add_data)
    with open(join("models", "SoccerNet", 'all_models_test_metrics_latex.txt'), 'w') as f:
        f.write(tabulate(data, headers=headers, tablefmt='latex', floatfmt=".4f"))


def release_gpu_memory():
    clear_session()
    gc.collect()


def saveSoccerNetSceneStarts(soccernet_path: str = "E:\\SoccerNet"):
    split = ["train", "test", "valid", "challenge"]
    for video in tqdm(getListGames(split)):
        base_path = join(soccernet_path, video)
        for half in range(1, 3):
            video_data = vread(join(base_path, f"{half}_224p.mkv"))
            scene_starts = scenedet(video_data)
            save(join(base_path, f"{half}_scene_starts"), scene_starts)


def get_closest_before(myList, myNumber):
    """
    From https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]

    return before, after


def saveIV3FeaturesForSoccerNet(soccernet_path: str = "E:\\SoccerNet", resnet: str = "models\\SoccerNet\\ResNet"
                                                                                     "\\checkpoints\\overall_best"
                                                                                     ".hdf5",
                                window_len: int = 5, stride: int = 1, fps: float = 3.2, batch_size: int = 32):
    iv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 398, 3), pooling="avg")

    chunk_size = window_len * 25
    for video in tqdm(getListGames(["train", "valid"])):
        base_path = join(soccernet_path, video)
        for half in range(1, 3):
            video_data = vread(join(base_path, f"{half}_224p.mkv"))
            num_chunks = video_data.shape[0] // chunk_size
            iv3_features = zeros((num_chunks, 16, 2048), dtype=single)
            for chunk in range(num_chunks):
                inputs = video_data[chunk * chunk_size: (chunk + 1) * chunk_size][::round(25 / fps)][:16].reshape(
                    (1, 16, 224, -1, 3))
                iv3_features[chunk, ...] = iv3.predict(preprocess_input(inputs.squeeze()), verbose=0)
            save(join(base_path, f"{half}_iv3_features"), iv3_features)

    for video in tqdm(getListGames(["test", "challenge"])):
        base_path = join(soccernet_path, video)
        for half in range(1, 3):
            video_data = vread(join(base_path, f"{half}_224p.mkv"))
            overlap_size = stride * 25
            num_chunks = (video_data.shape[0] - overlap_size) // (chunk_size - overlap_size)
            iv3_features = zeros((num_chunks, 16, 2048), dtype=single)
            for chunk in range(num_chunks):
                inputs = video_data[chunk * overlap_size: chunk * overlap_size + chunk_size][
                         ::round(25 / fps)][:16].reshape((1, 16, 224, -1, 3))
                iv3_features[chunk, ...] = iv3.predict(preprocess_input(inputs.squeeze()), verbose=0)
            save(join(base_path, f"{half}_iv3_features"), iv3_features)


def saveVGGFeaturesForSoccerNet(soccernet_path: str = "E:\\SoccerNet", window_len: int = 5, stride: int = 1):
    sound_model = VGGish(include_top=False, load_weights=True)

    x = sound_model.get_layer(name="conv4/conv4_2").output
    output_layer = GlobalAveragePooling2D()(x)
    sound_extractor = Model(inputs=sound_model.input, outputs=output_layer)

    chunk_size = window_len
    for video in tqdm(getListGames(["train", "valid"])):
        base_path = join(soccernet_path, video)
        for half in range(1, 3):
            video_data = VideoFileClip(join(base_path, f"{half}_224p.mkv"))
            audio = video_data.audio.to_soundarray()
            num_chunks = video_data.duration // chunk_size
            inputs = zeros((num_chunks, 16, 496, 64, 1), dtype=single)
            for chunk in range(num_chunks):
                audio_chunk = audio[int(video_data.audio.fps * chunk * chunk_size): int(
                    video_data.audio.fps * (chunk * chunk_size + chunk_size)), ...]
                audio_chunk_size = audio_chunk.shape[0] // 16
                for j in range(16):
                    inputs[chunk, j, ...] = preprocess_sound(
                        audio_chunk[j * audio_chunk_size: (j + 1) * audio_chunk_size, ...])
            vgg_features = sound_extractor.predict(inputs, verbose=0)
            save(join(base_path, f"{half}_VGG_features"), vgg_features)

    for video in tqdm(getListGames(["test", "challenge"])):
        base_path = join(soccernet_path, video)
        for half in range(1, 3):
            video_data = VideoFileClip(join(base_path, f"{half}_224p.mkv"))
            audio = video_data.audio.to_soundarray()
            overlap_size = stride
            num_chunks = (video_data.duration - overlap_size) // (chunk_size - overlap_size)
            inputs = zeros((num_chunks, 16, 496, 64, 1), dtype=single)
            for chunk in range(num_chunks):
                audio_chunk = audio[int(video_data.audio.fps * chunk * overlap_size): int(
                    video_data.audio.fps * (chunk * overlap_size + chunk_size)), ...]
                audio_chunk_size = audio_chunk.shape[0] // 16
                for j in range(16):
                    inputs[chunk, j, ...] = preprocess_sound(
                        audio_chunk[j * audio_chunk_size: (j + 1) * audio_chunk_size, ...])
            vgg_features = sound_extractor.predict(inputs, verbose=0)
            save(join(base_path, f"{half}_VGG_features"), vgg_features)


def saveResNetFeaturesForSoccerNet(soccernet_path: str = "E:\\SoccerNet", resnet: str = "models\\SoccerNet\\ResNet"
                                                                                        "\\checkpoints\\overall_best"
                                                                                        ".hdf5",
                                   window_len: int = 5, stride: int = 1, fps: float = 3.2, batch_size: int = 32):
    from data_generator import SoccerNetTestVideoGenerator

    resnet = load_model(resnet)
    resnet = Model(inputs=resnet.input, outputs=resnet.get_layer('flatten').output)

    for video in tqdm(getListGames(["test", "train", "valid"])):
        base_path = join(soccernet_path, video)
        for half in range(1, 3):
            if os.path.exists(join(base_path, f"{half}_ResNet_soccer_embeddings.npy")):
                continue
            generator = SoccerNetTestVideoGenerator(video, half, batch_size=batch_size, window_len=window_len, fps=fps,
                                                    data_path=soccernet_path, stride=stride)  # Test generator is
            # used because labels are not required and striding with feature frames may be useful for certain
            # architectures (e.g. transformers)
            resnet_features = resnet.predict(generator, workers=1, use_multiprocessing=False)
            save(join(base_path, f"{half}_ResNet_soccer_embeddings"), resnet_features)


def map_train_metrics_to_funcs(metrics):
    metrics = array(metrics, dtype=object)
    metrics[where(metrics == 'auc')[0]] = AUC(multi_label=True, num_labels=18)
    metrics[where(metrics == 'precision')[0]] = Precision(name='precision')
    metrics[where(metrics == 'recall')[0]] = Recall(name='recall')

    return list(metrics)


if __name__ == '__main__':
    # delete_soccernet_frames()
    # check_extract_soccernet_frames()
    for game in tqdm(getListGames(["train", "valid", "test", "challenge"])):
        source_path = join("F:\\SoccerNet", game)
        dest_path = join("C:\\Users\\pj\\Documents\\Masters\\EfficientSNSpotting\\SoccerNet", game)
        os.makedirs(dest_path, exist_ok=True)
        for half in range(1, 3):
            if not exists(join(dest_path, f"{half}_baidu_soccer_embeddings.h5")):
                with h5py.File(join(dest_path, f"{half}_baidu_soccer_embeddings.h5"), 'w') as h5f:
                    h5f.create_dataset('baidu', data=load(join(source_path, f"{half}_baidu_soccer_embeddings.npy")))
                os.remove(join(source_path, f"{half}_baidu_soccer_embeddings.npy"))
        if not exists(join(dest_path, f"Labels-cameras.json")) and exists(join(source_path, f"Labels-cameras.json")):
            shutil.copy(join(source_path, f"Labels-cameras.json"), join(dest_path, f"Labels-cameras.json"))
        if not exists(join(dest_path, f"Labels-v2.json")) and exists(join(source_path, f"Labels-v2.json")):
            shutil.copy(join(source_path, f"Labels-v2.json"), join(dest_path, f"Labels-v2.json"))
