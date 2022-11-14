import math
from abc import ABC
from math import ceil

from keras import Input, Model
from keras.activations import sigmoid
from keras.layers import Add, Flatten, Dense, Dropout
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding
from numpy import float32
import sys

from numpy.random import rand
from tensorflow import matmul, reshape
from tensorflow.python.keras.backend import softmax, sum as tf_sum, concatenate
from tensorflow.python.ops.nn_impl import normalize

sys.path.append("./keras-resnet3d-master/resnet3d")
from resnet3d import Resnet3DBuilder


def get_custom_objects():
    ret = {'TransformerEncoder': TransformerEncoder,
           'SinePositionEncoding': SinePositionEncoding}
    # for k, v in keras_transformer.get_custom_objects().items():
    #     ret[k] = v
    return ret


def createTransformerModel(model: str = "ResNet", seq_length: int = 7, frame_dim=8576):
    if "resnet" in model.lower():
        shape = (seq_length, frame_dim)
    elif "baidu" in model.lower():
        shape = (seq_length, frame_dim)
    inputs = Input(shape, dtype=float32)
    embedding = SinePositionEncoding()(inputs)
    outputs = Add()([inputs, embedding])

    for i in range(3):
        outputs = TransformerEncoder(intermediate_dim=64, num_heads=4)(outputs)

    outputs = Flatten()(outputs)
    # outputs = GlobalMaxPooling1D()(outputs)
    # outputs = Dropout(0.5)(outputs)
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
        model_ = createTransformerModel("baidu", seq_length=data["window length"],
                                        frame_dim=data["frame dims"][1] - data["frame dims"][0])
    elif "ann" in data["model"].lower():
        model_ = createANN(dataset=data["dataset"], seq_length=data["window length"], **data["model params"])
    elif "netvlad" in data["model"].lower():
        input = Input(shape=(int(data["window length"] * data["feature fps"]),
                             data["frame dims"][1] - data["frame dims"][0]), batch_size=data["batch size"])
        output = NetVLAD_PP(input_size=data["frame dims"][1] - data["frame dims"][0], window_size=data["window length"],
                            framerate=data["feature fps"])(input)
        model_ = Model(input, output)

    model_.summary()

    return model_


class NetVLAD(Model, ABC):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = (1 / math.sqrt(feature_size)) * rand(feature_size, cluster_size)
        self.clusters2 = (1 / math.sqrt(feature_size)) * rand(1, feature_size, cluster_size)

        self.add_batch_norm = add_batch_norm
        self.out_dim = cluster_size * feature_size

    def call(self, inputs, training=None, mask=None):
        max_samples = inputs.shape[1]

        if self.add_batch_norm:
            inputs = normalize(inputs, ord=2, axis=2)[0]

        inputs = reshape(inputs, [-1, self.feature_size])
        assignment = matmul(inputs, self.clusters)

        assignment = softmax(assignment, axis=1)
        assignment = reshape(assignment, (-1, max_samples, self.cluster_size))

        a_sum = tf_sum(assignment, axis=-2, keepdims=True)
        a = a_sum * self.clusters2

        assignment = reshape(assignment, (-1, assignment.shape[2], assignment.shape[1]))

        inputs = reshape(inputs, (-1, max_samples, self.feature_size))
        vlad = matmul(assignment, inputs)
        vlad = reshape(vlad, (-1, vlad.shape[2], vlad.shape[1]))
        vlad = vlad - a

        vlad = normalize(vlad)[0]

        vlad = reshape(vlad, (-1, self.cluster_size * self.feature_size))
        vlad = normalize(vlad)[0]

        return vlad


class NetVLAD_PP(Model, ABC):
    def __init__(self, weights=None, input_size=512, num_classes=17, vocab_size=64, window_size=15, framerate=2):
        super(NetVLAD_PP, self).__init__()

        self.window_size_frame = window_size * framerate
        self.input_size = input_size
        self.num_classes = num_classes
        self.framerate = framerate
        self.vlad_k = vocab_size

        if not self.input_size == 512:
            self.feature_extractor = Dense(512)
            self.input_size = 512

        self.pool_layer_before = NetVLAD(cluster_size=int(self.vlad_k / 2), feature_size=self.input_size,
                                         add_batch_norm=True)
        self.pool_layer_after = NetVLAD(cluster_size=int(self.vlad_k / 2), feature_size=self.input_size,
                                        add_batch_norm=True)
        self.fc = Dense(self.num_classes + 1)
        self.drop = Dropout(0.4)
        self.sigm = sigmoid

        if weights is not None:
            self.load_weights(weights)

    def call(self, inputs, training=None, mask=None):
        BS, FR, IC = inputs.shape

        if not IC == 512:
            inputs = reshape(inputs, (BS * FR, IC))
            inputs = self.feature_extractor(inputs)
            inputs = reshape(inputs, (BS, FR, -1))

        nb_frames_50 = int(inputs.shape[1] / 2)
        inputs_before_pooled = self.pool_layer_before(inputs[:, :nb_frames_50, :])
        inputs_after_pooled = self.pool_layer_after(inputs[:, nb_frames_50:, :])
        inputs_pooled = concatenate((inputs_before_pooled, inputs_after_pooled), axis=1)

        output = self.sigm(x=self.fc(self.drop(inputs_pooled)))

        return output


if __name__ == "__main__":
    vlad = NetVLAD(cluster_size=64, feature_size=512)

    feat_in = rand(3, 120, 512)
    print("in", feat_in.shape)
    feat_out = vlad(feat_in)
    print("out", feat_out.shape)
    print(512 * 64)

    BS = 256
    T = 15
    framerate = 2
    D = 512
    pool = "NetVLAD++"
    model = NetVLAD_PP(input_size=D, framerate=framerate, window_size=T)
    print(model)
    inp = rand(BS, T * framerate, D)
    print(inp.shape)
    output = model(inp)
    print(output.shape)
