import gc
import json
import os
from math import ceil

import cv2
import h5py
from tensorflow import shape, range, reduce_any, cast
from keras import Input
from keras.layers import Layer
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding
from numpy import float32, save, zeros, single, array
from os.path import join, exists
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D, Dense, Embedding, Flatten
from tqdm import tqdm
from keras.backend import clear_session
from tabulate import tabulate
from SoccerNet.Downloader import getListGames, SoccerNetDownloader
from skvideo.measure import scenedet
from skvideo.io import vread
from bisect import bisect_left
import sys

sys.path.append("./keras-resnet3d-master/resnet3d")
from resnet3d import Resnet3DBuilder

sys.path.append("./VGGish-master")

from vggish import VGGish
from preprocess_sound import preprocess_sound
from moviepy.editor import VideoFileClip
from tensorflow.config.experimental import list_physical_devices, set_memory_growth
from tensorflow.keras import mixed_precision
from os import environ
from tensorflow.config.optimizer import set_jit


def setup_environment(data):
    environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
    environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    gpus = list_physical_devices('GPU')
    for gpu in gpus:
        set_memory_growth(gpu, True)
    # mixed_precision.set_global_policy("mixed_float16")
    if data["jit"]:
        set_jit(True)

    release_gpu_memory()


def getDuration(video_path):
    """Get the duration (in seconds) for a video.

    Keyword arguments:
    video_path -- the path of the video
    """
    with VideoFileClip(video_path) as clip:
        return clip.reader.nframes


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
            if exists(join("E:\\SoccerNet", vid, f"{half}_frames.h5")):
                try:
                    with VideoFileClip(join("E:\\SoccerNet", vid, f"{half}_224p.mkv"), audio=False) as clip:
                        duration = clip.reader.nframes
                    with h5py.File(join("E:\\SoccerNet", vid, f"{half}_frames.h5"), 'r') as hf:
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

            with h5py.File(join("E:\\SoccerNet", vid, f"{half}_frames.h5"), 'w') as h5f:
                h5f.create_dataset('soccernet_3_125_fps', data=frames)


def get_cv_data(subset, cv_iter):
    """
    Read appropriate data split.
    """
    if cv_iter > 0:
        train_paths = getListGames(["train"])
        valid_paths = getListGames(["valid"])
        test_paths = getListGames(["test"])
        if subset == "train":
            if cv_iter == 1:
                ret = train_paths[100:]
                ret.extend(valid_paths)
            elif cv_iter == 2:
                ret = train_paths[200:]
                ret.extend(valid_paths)
                ret.extend(test_paths)
            elif cv_iter == 3:
                ret = valid_paths
                ret.extend(test_paths)
                ret.extend(train_paths[:100])
            elif cv_iter == 4:
                ret = test_paths
                ret.extend(train_paths[:200])
        if subset == "valid":
            if cv_iter == 1:
                ret = test_paths
            elif cv_iter == 2:
                ret = train_paths[:100]
            elif cv_iter == 3:
                ret = train_paths[100:200]
            elif cv_iter == 4:
                ret = train_paths[200:300]
        if subset == "test":
            if cv_iter == 1:
                ret = train_paths[:100]
            elif cv_iter == 2:
                ret = train_paths[100:200]
            elif cv_iter == 3:
                ret = train_paths[200:300]
            elif cv_iter == 4:
                ret = valid_paths
        return ret
    else:
        return getListGames([subset])


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


def save_train_latex_table(dataset: str = "SoccerNet"):
    base_len = len(join("models", dataset)) + 1
    headers = ['Model']
    data = [["Training Accuracy"], ["Validation Accuracy"], ["Test Accuracy"],
            ["Training Loss"], ["Validation Loss"], ["Epochs"]]
    for base, dirs, files in os.walk(join("models", dataset)):
        for file in files:
            if file == "avg_metrics_all_cv.json":
                path = join(base, file)
                with open(path, 'r') as f:
                    jdata = json.load(f)
                    headers.append(f'{path[base_len:-len(file) - len("results") - len("CV") - 3]}')
                    data[0].append(f'{jdata["train acc"]:.4f} \u00B1 {jdata["train acc std"]:.4f}')
                    data[1].append(f'{jdata["valid acc"]:.4f} \u00B1 {jdata["valid acc std"]:.4f}')
                    data[2].append(f'{jdata["test acc"]:.4f} \u00B1 {jdata["test acc std"]:.4f}')
                    data[3].append(f'{jdata["train loss"]:.4f} \u00B1 {jdata["train loss std"]:.4f}')
                    data[4].append(f'{jdata["valid loss"]:.4f} \u00B1 {jdata["valid loss std"]:.4f}')
                    data[5].append(f'{jdata["epochs"]:.4f} \u00B1 {jdata["epochs std"]:.4f}')

    with open(join("models", dataset, 'all_models_train_metrics_latex.txt'), 'w') as f:
        f.write(tabulate(data, headers=headers, tablefmt='latex', floatfmt=".4f"))


def save_test_latex_table(dataset: str = "SoccerNet"):
    base_len = len(join("models", dataset)) + 1
    headers = ['Model/Metric'] + [f'Class {i}' for i in range(17)]
    data = []
    for base, dirs, files in os.walk(join("models", dataset)):
        for file in files:
            if file == "results.json":
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
    with open(join("models", dataset, 'all_models_test_metrics_latex.txt'), 'w') as f:
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


class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = shape(inputs)[1]
        positions = range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        embedded_positions = SinePositionEncoding()(embedded_positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = reduce_any(cast(inputs, "bool"), axis=-1)
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config


def createTransformerModel(model: str = "ResNet", seq_length: int = 7):
    if "resnet" in model.lower():
        shape = (seq_length, 512)
    elif "baidu" in model.lower():
        shape = (seq_length, 8576)
    inputs = Input(shape, dtype=float32)
    outputs = PositionalEmbedding(shape[0], shape[1])(inputs)

    for i in range(3):
        outputs = TransformerEncoder(intermediate_dim=64, num_heads=4)(outputs)

    outputs = Flatten()(outputs)
    outputs = Dense(18, activation='sigmoid')(outputs)

    model = Model(inputs, outputs)

    return model


def createANN(dataset: str = "CSports", seq_length: int = 7, num_layers: int = 2, num_nodes: int = 512):
    if "baidu" in dataset.lower():
        shape = (seq_length, 8576)
    elif "resnet" in dataset.lower():
        shape = (seq_length, 512)

    inputs = Input(shape, dtype=float32)
    outputs = Flatten()(inputs)

    for i in range(num_layers):
        outputs = Dense(num_nodes)(outputs)

    outputs = Dense(18, activation='sigmoid')(outputs)

    model = Model(inputs, outputs)

    return model


def create_model(data):
    if "resnet" in data["model"].lower():
        model_ = Resnet3DBuilder.build_resnet_18((int(ceil(data["window length"] * data["feature fps"])),
                                                  data["frame dims"][0], data["frame dims"][1], 3), 18,
                                                 multilabel=True)
    elif "baidu" in data["model"].lower():
        model_ = createTransformerModel("baidu", seq_length=data["window length"])
    elif "ann" in data["model"].lower():
        model_ = createANN(dataset=data["dataset"], seq_length=data["window length"], **data["model params"])

    model_.summary()

    return model_


if __name__ == '__main__':
    # delete_soccernet_frames()
    check_extract_soccernet_frames()
