from multiprocessing import Pool, cpu_count
import h5py
from keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from numpy import ceil, single, zeros, reshape, divide, load, uint8, delete, where, arange, \
    stack
from numpy.random import randint, choice
from os.path import join
from SoccerNet.DataLoader import getDuration
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
import json
from sklearn.utils import shuffle
from util import resize, get_cv_data


class SoccerNetTrainDataset:
    def __init__(self, train: str = "train", data_path: str = 'E:\\SoccerNet', fps: float = 3.2,
                 window_len: int = 5, cv_iter: int = 0, data_fraction=1):
        self.y = None
        self.sample_start_times = None
        self.window_len = window_len
        self.labels = "Labels-v2.json"
        self.train = train

        self.all_vid_paths = get_cv_data(train, cv_iter, data_fraction)

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
                 window_len: int = 5, cv_iter: int = 0, frame_dims=(224, 398), resize_method="", data_fraction=1):
        """
        Initialise the class.
        :param train: Whether 'train', 'validation' or 'test' data should be loaded.
        :param data_path: The path which contains the video clip folders.
        :param batch_size: Number of samples in each generated batch.
        """
        super().__init__(train, data_path, fps, window_len, cv_iter, data_fraction)
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


class DeepFeatureGenerator:
    def __init__(self, data, cv_iter=0, data_subset="train"):
        self.window_len = data["window length"]
        if data_subset != "test":
            self.stride = data["stride"]
        else:
            self.stride = data["test stride"]
        self.base_path = data["dataset path"]
        self.feature_type = data["features"]

        self.data_subset = data_subset
        self.feature_paths = []
        for vid in get_cv_data(data_subset, cv_iter, data["data fraction"]):
            for half in range(1, 3):
                self.feature_paths.append((join(self.base_path, vid), f"{half}"))
        self.replays = []
        if self.data_subset != "test":
            self.feature_paths = shuffle(self.feature_paths)
            if data["remove replays"]:
                for i in range(len(self.feature_paths)):
                    self.replays.append(self.get_replays(i))
        self.num_classes = 17
        self.label_file = "Labels-v2.json"
        self.dict_event = EVENT_DICTIONARY_V2
        self.fps = data["feature fps"]
        self.remove_replays = data["remove replays"]
        self.balance_classes = data["balance classes"]
        self.frame_dims = data["frame dims"]
        self.overlap = self.window_len - self.stride

    def __data_generation(self, index):
        if self.feature_type.split('.')[1] == "npy":
            X = load(join(self.feature_paths[index][0], self.feature_paths[index][1] + "_" + self.feature_type))[:, self.frame_dims[0]:self.frame_dims[1]]
        elif self.feature_type.split('.')[1] == "h5":
            with h5py.File(join(self.feature_paths[index][0], self.feature_paths[index][1] + "_" + self.feature_type), 'r') as hf:
                X = hf['baidu'][:, self.frame_dims[0]:self.frame_dims[1]]
        idx = arange(start=0, stop=X.shape[0] - self.window_len, step=self.stride)
        idxs = []
        for i in arange(0, self.window_len):
            idxs.append(idx + i)
        idx = stack(idxs, axis=1)

        X = X[idx, ...]

        assert X.shape[1] == self.window_len
        assert X.shape[2] == self.frame_dims[1] - self.frame_dims[0]

        if self.data_subset != "test":
            Y = zeros((X.shape[0], 1))
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
                frame = int(self.fps * t_seconds) // self.stride

                if frame >= X.shape[0]:
                    continue

                label = self.dict_event[event]  # event label
                start = max(frame - self.overlap, 0)
                Y[start: frame + 1, 0] = label + 1

            if self.remove_replays:
                for replay in self.replays[index]:
                    start, end, half = replay[0], replay[1], replay[2]
                    start, end = int(start * self.fps) // self.stride, int(end * self.fps) // self.stride
                    X = delete(X, list(range(start, end + 1)), axis=0)
                    Y = delete(Y, list(range(start, end + 1)), axis=0)

            if self.balance_classes:
                delete_candidates = where(Y[:, 0] == 0)[0]
                desired_num_bg = int((X.shape[0] - len(delete_candidates)) / 17)
                delete_candidates = choice(delete_candidates, size=len(delete_candidates) - desired_num_bg,
                                           replace=False)
                X = delete(X, delete_candidates, axis=0)
                Y = delete(Y, delete_candidates, axis=0)

            return X, to_categorical(Y, num_classes=18)

        return X

    def get_replays(self, index):
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
        return replays

    def __call__(self, *args, **kwargs):
        if self.data_subset != "test":
            for i in range(len(self.feature_paths)):
                X, Y = self.__data_generation(i)
                for sample in range(X.shape[0]):
                    yield X[sample, ...], \
                          Y[sample, ...]
        else:
            for i in range(len(self.feature_paths)):
                yield self.__data_generation(i)


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
