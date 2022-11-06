from multiprocessing import Pool, cpu_count
import h5py
from tensorflow.keras.utils import Sequence
from numpy import ceil, single, zeros, reshape, divide, load, array, clip, array_split, uint8, delete
from numpy import sum as np_sum
from numpy.random import randint, choice
from os.path import join
from SoccerNet.DataLoader import getDuration
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
import json
from sklearn.utils import shuffle
from util import resize, get_cv_data


class SoccerNetTrainDataset:
    def __init__(self, train: str = "train", data_path: str = 'E:\\SoccerNet', fps: float = 3.2,
                 window_len: int = 5, cv_iter: int = 0):
        self.y = None
        self.sample_start_times = None
        self.window_len = window_len
        self.labels = "Labels-v2.json"
        self.train = train

        self.all_vid_paths = get_cv_data(train, cv_iter)

        self.base_data_path = data_path

        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17

        self.reset_epoch()

        self.train = train

        self.FPS = fps

    def reset_epoch(self):
        print("Resetting generator")
        self.all_vid_paths = shuffle(self.all_vid_paths)
        self.sample_start_times = []
        self.y = []

        with Pool() as pool:
            for start_tuples, labels in pool.imap(self.get_vid_samples, self.all_vid_paths,
                                                  chunksize=int(len(self.all_vid_paths) / cpu_count())):
                [self.y.append(labels[i]) for i in range(len(labels))]
                [self.sample_start_times.append(start_tuples[i]) for i in range(len(start_tuples))]

        self.y = reshape(self.y, (-1, self.num_classes + 1)).astype(uint8)
        self.y, self.sample_start_times = shuffle(self.y, self.sample_start_times)

    def get_vid_samples(self, vid):
        y = []
        sample_start_times = []

        labels = json.load(open(join(self.base_data_path, vid, self.labels)))
        half1_duration = getDuration(join(self.base_data_path, vid, "1_224p.mkv"))
        half2_duration = getDuration(join(self.base_data_path, vid, "2_224p.mkv"))
        num_vid_samples = 0
        camera_labels = json.load(open(join(self.base_data_path, vid, "Labels-cameras.json")))
        replays = []
        for annotation in enumerate(camera_labels["annotations"]):
            if annotation[1]["replay"] != "replay":
                continue
            time = camera_labels["annotations"][annotation[0] - 1]["gameTime"]
            read_half = int(time[0])
            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            start = seconds + 60 * minutes  # time of event in seconds
            time = annotation[1]["gameTime"]
            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            end = seconds + 60 * minutes  # time of event in seconds
            replays.append((start, end, read_half))
        for annotation in labels["annotations"]:
            if annotation["visibility"] == "not shown":
                continue
            time = annotation["gameTime"]
            read_half = int(time[0])
            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            t_seconds = seconds + 60 * minutes  # time of event in seconds

            event = annotation["label"]
            if event not in self.dict_event:
                continue
            label = self.dict_event[event]  # event label
            # if the event falls close enough to the previous event such that there is no label annotation
            # uncertainty, group the events into one label
            if len(y) > 0 and abs(
                    t_seconds - sample_start_times[-1][0]) <= self.window_len - 1 and read_half == \
                    sample_start_times[-1][1] and vid == sample_start_times[-1][
                2]:  # buffer of 1 s on each
                # side of window
                y[-1][label + 1] = 1
            else:
                new_label = zeros((self.num_classes + 1,))
                new_label[label + 1] = 1
                y.append(new_label)
                num_vid_samples += 1
                if t_seconds < (half1_duration if read_half == 1 else half2_duration) - self.window_len:
                    sample_start_times.append((randint(max(t_seconds - (self.window_len - 1), 0),
                                                       max(t_seconds, 1)), read_half, vid))
                else:
                    sample_start_times.append(((half1_duration if read_half == 1 else half2_duration)
                                               - self.window_len, read_half, vid))
        # Add BG samples
        num_bg_samples = int(ceil(num_vid_samples / self.num_classes))
        temp_start_times = []
        for i in range(num_bg_samples):
            # choose a random  valid time in-between samples
            new_choice, half, vid_ = self.find_bg_sample(num_vid_samples, half1_duration, half2_duration,
                                                         replays, sample_start_times)

            temp_start_times.append((new_choice, half, vid_))
            new_label = zeros((self.num_classes + 1,))
            new_label[0] = 1
            y.append(new_label)
        [sample_start_times.append(temp_start_times[i]) for i in range(len(temp_start_times))]

        return sample_start_times, y

    def find_bg_sample(self, num_vid_samples, half1_duration, half2_duration, replays, sample_start_times):
        choice_index = randint(len(sample_start_times) - num_vid_samples, len(sample_start_times))
        before = choice([True, False])
        if before:
            # if there is no sample before:
            if choice_index == 0 or (
                    sample_start_times[choice_index][1] != sample_start_times[choice_index - 1][1]):
                if sample_start_times[choice_index][0] >= self.window_len:  # if a sample can fit before
                    new_choice = randint(0, sample_start_times[choice_index][0] - self.window_len + 1)
                else:
                    return self.find_bg_sample(num_vid_samples, half1_duration, half2_duration, replays,
                                               sample_start_times)
            else:  # there is a sample before
                if sample_start_times[choice_index][0] - sample_start_times[choice_index - 1][0] - \
                        self.window_len >= self.window_len:  # a sample can fit in between the two samples
                    new_choice = randint(sample_start_times[choice_index - 1][0] + self.window_len,
                                         sample_start_times[choice_index][0] - self.window_len + 1)
                else:
                    return self.find_bg_sample(num_vid_samples, half1_duration, half2_duration, replays,
                                               sample_start_times)
        else:  # find bg frame after choice
            # if there is no sample after
            if choice_index == len(sample_start_times) - 1 or (
                    sample_start_times[choice_index][1] != sample_start_times[choice_index + 1][1]):
                if sample_start_times[choice_index][0] + self.window_len <= (
                        half1_duration if sample_start_times[choice_index][
                                              1] == 1 else half2_duration) - self.window_len:  # if a sample
                    # can fit after
                    new_choice = randint(sample_start_times[choice_index][0] + self.window_len, (
                        half1_duration if sample_start_times[choice_index][
                                              1] == 1 else half2_duration) - self.window_len + 1)
                else:
                    return self.find_bg_sample(num_vid_samples, half1_duration, half2_duration, replays,
                                               sample_start_times)
            else:
                # there is a sample after
                if sample_start_times[choice_index + 1][0] - sample_start_times[choice_index][
                    0] - self.window_len >= self.window_len:  # a sample can fit in between the two samples
                    new_choice = randint(sample_start_times[choice_index][0] + self.window_len,
                                         sample_start_times[choice_index + 1][0] - self.window_len + 1)
                else:
                    return self.find_bg_sample(num_vid_samples, half1_duration, half2_duration, replays,
                                               sample_start_times)
        if new_choice >= (half1_duration if sample_start_times[choice_index][1] == 1 else half2_duration):
            raise Exception(
                f"Somehow selected start time {new_choice} from half {sample_start_times[choice_index][1]} of "
                f"{sample_start_times[choice_index][2]} which has duration {half1_duration if sample_start_times[choice_index][1] == 1 else half2_duration},"
                f" {'after' if not before else 'before'} sample with start time {sample_start_times[choice_index][0]}.")
        for replay in replays:
            if replay[0] < new_choice + self.window_len <= replay[1] and sample_start_times[choice_index][1] == \
                    replay[2]:
                return self.find_bg_sample(num_vid_samples, half1_duration, half2_duration, replays, sample_start_times)
            if replay[0] <= new_choice < replay[1] and sample_start_times[choice_index][1] == replay[2]:
                return self.find_bg_sample(num_vid_samples, half1_duration, half2_duration, replays, sample_start_times)
        return new_choice, sample_start_times[choice_index][1], sample_start_times[choice_index][2]

    def __call__(self, *args, **kwargs):
        for i in range(len(self.y)):
            start, half, vid = self.sample_start_times[i]
            with h5py.File(join(self.base_data_path, vid, f'{half}_frames.h5'), 'r') as hf:
                frames = hf['soccernet_3_125_fps'][int(start * self.FPS):
                                                   int(start * self.FPS) +
                                                   int(ceil(self.window_len * self.FPS)), ...]
                yield frames, self.y[i]
        self.reset_epoch()


class SoccerNetTrainVideoDataGenerator(Sequence, SoccerNetTrainDataset):
    """
    Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Class to generate train data for training the 3D ResNet branch since all the training data can't fit into
    memory.
    Inherits from Sequence to allow for multiprocessing.
    """

    def __init__(self, batch_size: int = 32, train: str = "train", data_path: str = 'E:\\SoccerNet', fps: float = 3.2,
                 window_len: int = 5, cv_iter: int = 0, frame_dims=(224, 398), resize_method=""):
        """
        Initialise the class.
        :param train: Whether 'train', 'validation' or 'test' data should be loaded.
        :param data_path: The path which contains the video clip folders.
        :param batch_size: Number of samples in each generated batch.
        """
        super().__init__(train, data_path, fps, window_len, cv_iter)
        self.batch_size = batch_size
        self.frame_dims = frame_dims
        self.resize_method = resize_method

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(ceil(self.y.shape[0] / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate data
        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):
        self.reset_epoch()

    def get_sample(self, start, half, vid):
        with h5py.File(join(self.base_data_path, vid, f'{half}_frames.h5'), 'r') as hf:
            frames = hf['soccernet_3_125_fps'][int(start * self.FPS):
                                               int(start * self.FPS) +
                                               int(ceil(self.window_len * self.FPS)), ...]

            frames = resize(frames, self.resize_method, self.frame_dims)

        return divide(frames, 255.)

    def __data_generation(self, index):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)

        y = self.y[index * self.batch_size: (index + 1) * self.batch_size, :]
        start_times = self.sample_start_times[index * self.batch_size: (index + 1) * self.batch_size]
        samples = zeros((y.shape[0], int(ceil(self.window_len * self.FPS)), self.frame_dims[0], self.frame_dims[1], 3),
                        dtype=single)

        for label in range(y.shape[0]):
            samples[label, ...] = self.get_sample(start_times[label][0], start_times[label][1], start_times[label][2])

        return samples, y


class TransformerTrainFeatureGenerator:
    def __init__(self, window_len: int = 7, stride: int = 7, base_path: str = "E:\\SoccerNet",
                 feature_type: str = "baidu", data_subset: str = "train", extraction_window=5,
                 extraction_stride: int = 1, cv_iter=0):
        self.window_len = window_len
        self.stride = stride
        self.base_path = base_path
        self.feature_type = f"{feature_type}_soccer_embeddings.npy"

        self.data_subset = data_subset
        self.feature_paths = []
        for vid in get_cv_data(data_subset, cv_iter):
            for half in range(1, 3):
                self.feature_paths.append((join(self.base_path, vid), f"{half}"))
        self.feature_paths = shuffle(self.feature_paths)
        self.num_classes = 17
        self.label_file = "Labels-v2.json"
        self.dict_event = EVENT_DICTIONARY_V2
        self.extractor_win = extraction_window
        self.extractor_stride = extraction_stride  # assumed

    def __data_generation(self, index):
        X = load(join(self.feature_paths[index][0], self.feature_paths[index][1] + "_" + self.feature_type))
        Y = zeros((X.shape[0], self.num_classes + 1))
        Y[:, 0] = 1

        camera_labels = json.load(open(join(self.feature_paths[index][0], "Labels-cameras.json")))
        replays = []
        for annotation in enumerate(camera_labels["annotations"]):
            if annotation[1]["replay"] != "replay":
                continue
            time = camera_labels["annotations"][annotation[0] - 1]["gameTime"]
            read_half = int(time[0])
            if read_half != self.feature_paths[index][1]:
                continue
            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            start = seconds + 60 * minutes  # time of event in seconds
            time = annotation[1]["gameTime"]
            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            end = seconds + 60 * minutes  # time of event in seconds
            replays.append((start, end, read_half))

        labels = json.load(open(join(self.feature_paths[index][0], "Labels-v2.json")))
        for annotation in labels["annotations"]:
            # if annotation["visibility"] == "not shown":
            #     continue
            time = annotation["gameTime"]
            read_half = time[0]
            if read_half != self.feature_paths[index][1]:
                continue
            event = annotation["label"]
            if event not in self.dict_event:
                continue
            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            t_seconds = seconds + 60 * minutes  # time of event in seconds

            if t_seconds < self.extractor_win - self.extractor_stride:
                f_index = 0
                rest = max(t_seconds // self.extractor_stride, 1)
            else:
                f_index = (t_seconds - (self.extractor_win - self.extractor_stride)) // self.extractor_stride
                rest = (self.extractor_win - self.extractor_stride) // self.extractor_stride

            label = self.dict_event[event]  # event label

            Y[f_index: f_index + rest, label + 1] = 1
            Y[f_index: f_index + rest, 0] = 0

        frames_per_sample = self.window_len
        if self.stride == self.window_len:
            num_to_discard = X.shape[0] % frames_per_sample
            if num_to_discard > 0:
                X = X[:-num_to_discard, ...]
                Y = Y[:-num_to_discard, ...]
            assert X.shape[0] % frames_per_sample == 0
            num_samples = X.shape[0] // frames_per_sample
            X = reshape(array_split(X, num_samples), (num_samples, frames_per_sample, -1))
            Y = clip(
                np_sum(reshape(array_split(Y, num_samples), (num_samples, frames_per_sample, self.num_classes + 1)),
                       axis=1), 0, 1)
        elif self.stride < self.window_len:
            temp = []
            temp_y = []
            num_chunks = (X.shape[0] - (frames_per_sample - self.stride)) // self.stride
            for chunk in range(num_chunks):
                temp.append(X[chunk * self.stride: chunk * self.stride + self.window_len, ...])
                temp_y.append(
                    clip(np_sum(Y[chunk * self.stride: chunk * self.stride + self.window_len, ...], axis=0), 0,
                         1))
            X = array(temp)
            Y = array(temp_y)

        for row in range(Y.shape[0]):
            if any(Y[row, 1:] == 1):
                Y[row, 0] = 0

        for replay in replays:
            start, end, half = replay[0], replay[1], replay[2]
            f_index = (start - (self.extractor_win - self.extractor_stride)) // self.extractor_stride
            l_index = (end - (self.extractor_win - self.extractor_stride)) // self.extractor_stride
            rest = (self.extractor_win - self.extractor_stride) // self.extractor_stride

            X = delete(X, list(range(f_index, l_index + rest + 1)), axis=0)
            Y = delete(Y, list(range(f_index, l_index + rest + 1)), axis=0)

        if self.data_subset == "test":
            return X
        else:
            return shuffle(X, Y)

    def __call__(self, *args, **kwargs):
        for i in range(len(self.feature_paths)):
            X, Y = self.__data_generation(i)
            for sample in range(X.shape[0]):
                if self.data_subset != "test":
                    yield X[sample, ...], \
                          Y[sample, ...]
                else:
                    yield X[sample, ...]


class SoccerNetTestVideoGenerator(Sequence):
    def __init__(self, video: str, half: int, batch_size: int = 32, data_path: str = "E:\\SoccerNet", fps: float = 3.2,
                 window_len: int = 5, stride: int = 1, frame_dims=(224, 398), resize_method=""):
        """
        Initialise the class.
        :param data_path: The path which contains the video clip folders.
        :param batch_size: Number of samples in each generated batch.
        """
        self.batch_size = batch_size
        self.window_len = window_len
        self.stride = stride
        self.num_classes = 17
        """
        Read appropriate data split.
        """
        self.base_data_path = data_path

        self.FPS = fps

        self.half = half

        self.video = video

        with h5py.File(join(self.base_data_path,
                            self.video, f'{self.half}_frames.h5'), 'r') as hf:
            self.vid_frames = hf['soccernet_3_125_fps'][:]

            self.vid_frames = resize(self.vid_frames, resize_method, frame_dims)

        self.frame_dims = frame_dims

        self.num_samples = (self.vid_frames.shape[0] - (self.window_len - self.stride) * fps) // (self.stride * fps)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate data
        X = self.__data_generation(index)

        return X

    def __data_generation(self, index):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        num_samples = min(self.num_samples - index * self.batch_size, self.batch_size)

        samples = zeros((num_samples, int(ceil(self.window_len * self.FPS)), self.frame_dims[0], self.frame_dims[1], 3),
                        dtype=single)

        for label in range(num_samples):
            start = (index * self.batch_size + label) * self.stride
            frames = self.vid_frames[int(start * self.FPS):
                                     int(start * self.FPS) +
                                     int(ceil(self.window_len * self.FPS)), ...]

            samples[label, ...] = divide(frames, 255.)

        return samples
