from keras.layers import Input, LSTM, Embedding, GlobalMaxPooling1D, Lambda, subtract, Conv2D, Dense, Activation
from keras.layers.merge import Dot, Subtract, Add, Multiply
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.initializers import glorot_uniform, Orthogonal
from keras import losses
from keras import backend as K
import tensorflow as tf
import numpy as np
from deeppavlov.core.models.tf_backend import TfModelMeta
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.layers import keras_layers
from pathlib import Path
from deeppavlov.core.commands.utils import expand_path
from typing import List, Callable
from deeppavlov.core.models.keras_model import KerasModel

log = get_logger(__name__)

class SiameseNetwork(metaclass=TfModelMeta):

    """Class to perform context-response matching with neural networks.

    Args:
        toks_num: A size of `tok2int` vocabulary to build embedding layer.
        chars_num: A size of `char2int` vocabulary to build character-level embedding layer.

        learning_rate: Learning rate.
        device_num: A number of a device to perform model training on if several devices are available in a system.
        seed: Random seed.
        shared_weights: Whether to use shared weights in the model to encode contexts and responses.
        triplet_loss: Whether to use a model with triplet loss.
            If ``False``, a model with crossentropy loss will be used.
        margin: A margin parameter for triplet loss. Only required if ``triplet_mode`` is set to ``True``.
        token_embeddings: Whether to use token (word) embeddings in the model.
        max_sequence_length: A maximum length of a sequence in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        tok_dynamic_batch:  Whether to use dynamic batching. If ``True``, a maximum length of a sequence for a batch
            will be equal to the maximum of all sequences lengths from this batch,
            but not higher than ``max_sequence_length``.
        embedding_dim: Dimensionality of token (word) embeddings.
        char_embeddings: Whether to use character-level token (word) embeddings in the model.
        max_token_length: A maximum length of a token for representing it by a character-level embedding.
        char_dynamic_batch: Whether to use dynamic batching for character-level embeddings.
            If ``True``, a maximum length of a token for a batch
            will be equal to the maximum of all tokens lengths from this batch,
            but not higher than ``max_token_length``.
        char_emb_dim: Dimensionality of character-level embeddings.
        reccurent: A type of the RNN cell. Possible values are ``lstm`` and ``bilstm``.
        hidden_dim: Dimensionality of the hidden state of the RNN cell. If ``reccurent`` equals ``bilstm``
            to get the actual dimensionality ``hidden_dim`` should be doubled.
        max_pooling: Whether to use max-pooling operation to get context (response) vector representation.
            If ``False``, the last hidden state of the RNN will be used.
        hard_triplets_sampling: Whether to use hard triplets sampling to train the model
            i.e. to choose negative samples close to positive ones.
        hardest_positives: Whether to use only one hardest positive sample per each anchor sample.
            It is only used when ``hard_triplets_sampling`` is set to ``True``.
        semi_hard_negatives: Whether hard negative samples should be further away from anchor samples
            than positive samples or not. It is only used when ``hard_triplets_sampling`` is set to ``True``.
        num_hardest_negatives: It is only used when ``hard_triplets_sampling`` is set to ``True``
            and ``semi_hard_negatives`` is set to ``False``.
    """

    def __init__(self,
                 save_path: str,
                 load_path: str,
                 emb_matrix: np.ndarray = None,
                 len_char_vocab: int = None,
                 learning_rate: float = 1e-3,
                 device_num: int = 0,
                 seed: int = None,
                 shared_weights: bool = True,
                 triplet_loss: bool = True,
                 margin: float = 0.1,
                 token_embeddings: bool = True,
                 embedding_dim: int = 300,
                 char_embeddings: bool = False,
                 char_emb_dim: int = 32,
                 highway_on_top: bool = False,
                 reccurent: str = "bilstm",
                 hidden_dim: int = 300,
                 max_pooling: bool = True,
                 hard_triplets_sampling: bool = False,
                 hardest_positives: bool = False,
                 semi_hard_negatives: bool = False,
                 num_hardest_negatives: int = None,
                 network: Callable = "bilstm_nn",
                 **kwargs):

        self.save_path = expand_path(save_path).resolve()
        self.load_path = expand_path(load_path).resolve()
        self.emb_matrix = emb_matrix
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.device_num = device_num
        self.shared_weights = shared_weights
        self.pooling = max_pooling
        self.recurrent = reccurent
        self.token_embeddings = token_embeddings
        self.char_embeddings = char_embeddings
        self.chars_num = len_char_vocab
        self.char_emb_dim = char_emb_dim
        self.highway_on_top = highway_on_top
        self.hard_triplets_sampling = hard_triplets_sampling
        self.hardest_positives = hardest_positives
        self.semi_hard_negatives = semi_hard_negatives
        self.num_hardest_negatives = num_hardest_negatives
        self.triplet_mode = triplet_loss

        self.sess = self._config_session()
        K.set_session(self.sess)

        self.optimizer = Adam(lr=self.learning_rate)
        self.embeddings = network.embeddings_model()
        # self.embeddings = self.simple_model()
        if self.triplet_mode:
            self.loss = self.triplet_loss
        else:
            self.loss = losses.binary_crossentropy
        self.obj_model = self.loss_model()
        self.obj_model.compile(loss=self.loss, optimizer=self.optimizer)
        self.score_model = self.prediction_model()
        self.context_embedding = Model(inputs=self.embeddings.inputs,
                                 outputs=self.embeddings.outputs[0])
        self.response_embedding = Model(inputs=self.embeddings.inputs,
                                 outputs=self.embeddings.outputs[1])

        if self.load_path.exists():
           self.load()
        else:
            self.load_initial_emb_matrix()

    def simple_model(self):
        context = Input(shape=(200,))
        response = Input(shape=(200,))
        model = Model([context, response], [context, response])
        return model

    def _config_session(self):
        """
        Configure session for particular device
        Returns:
            tensorflow.Session
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(self.device_num)
        return tf.Session(config=config)

    def load(self):
        log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
        self.obj_model.load_weights(str(self.load_path))

    def save(self):
        log.info("[saving `{}`]".format(self.__class__.__name__))
        self.obj_model.save_weights(str(self.save_path))
        self.context_embedding.save(str(self.save_path.parent / 'sen_emb_model.h5'))

    def load_initial_emb_matrix(self):
        log.info("[initializing new `{}`]".format(self.__class__.__name__))
        if self.emb_matrix is not None:
            if self.token_embeddings and not self.char_embeddings:
                if self.shared_weights:
                    self.embeddings.get_layer(name="embedding").set_weights([self.emb_matrix])
                else:
                    self.embeddings.get_layer(name="embedding_a").set_weights([self.emb_matrix])
                    self.embeddings.get_layer(name="embedding_b").set_weights([self.emb_matrix])

    def prediction_model(self):
        c_shape = K.int_shape(self.embeddings.inputs[0])
        r_shape = K.int_shape(self.embeddings.inputs[1])
        c = Input(batch_shape=c_shape)
        r = Input(batch_shape=r_shape)
        emb_c, emb_r = self.embeddings([c, r])
        # if self.distance == "cos_similarity":
        #     cosine_layer = Dot(normalize=True, axes=-1, name="score_model")
        #     score = cosine_layer([emb_c, emb_r])
        #     score = Lambda(lambda x: 1. - x)(score)
        if self.triplet_mode:
            dist_score = Lambda(lambda x: K.expand_dims(self.euclidian_dist(x)), name="score_model")
            score = dist_score([emb_c, emb_r])
        else:
            dist = Lambda(self.diff_mult_dist)([emb_c, emb_r])
            score = Dense(1, activation='sigmoid', name="score_model")(dist)
            score = Lambda(lambda x: 1. - x)(score)
        score = Lambda(lambda x: 1. - x)(score)
        model = Model([c, r], score)
        return model

    def loss_model(self):
        c_shape = K.int_shape(self.embeddings.inputs[0])
        r_shape = K.int_shape(self.embeddings.inputs[1])
        c = Input(batch_shape=c_shape)
        r = Input(batch_shape=r_shape)
        emb_c, emb_r = self.embeddings([c, r])
        if self.triplet_mode:
            dist = Lambda(self._pairwise_distances)([emb_c, emb_r])
        else:
            dist = Lambda(self.diff_mult_dist)([emb_c, emb_r])
            dist = Dense(1, activation='sigmoid', name="score_model")(dist)
        model = Model([c, r], dist)
        return model

    def diff_mult_dist(self, inputs):
        input1, input2 = inputs
        a = K.abs(input1-input2)
        b = Multiply()(inputs)
        return K.concatenate([input1, input2, a, b])

    def euclidian_dist(self, x_pair):
        x1_norm = K.l2_normalize(x_pair[0], axis=1)
        x2_norm = K.l2_normalize(x_pair[1], axis=1)
        diff = x1_norm - x2_norm
        square = K.square(diff)
        sum = K.sum(square, axis=1)
        sum = K.clip(sum, min_value=1e-12, max_value=None)
        dist = K.sqrt(sum)
        return dist

    def _pairwise_distances(self, inputs):
        emb_c, emb_r = inputs
        dot_product = K.dot(emb_c, K.transpose(emb_r))
        square_norm = K.batch_dot(emb_c, emb_r, axes=1)
        distances = K.transpose(square_norm) - 2.0 * dot_product - square_norm
        distances = K.clip(distances, 0.0, None)
        mask = K.cast(K.equal(distances, 0.0), K.dtype(distances))
        distances = distances + mask * 1e-16
        distances = K.sqrt(distances)
        distances = distances * (1.0 - mask)
        return distances

    def triplet_loss(self, y_true, pairwise_dist):
        """Triplet loss function"""
        if self.hard_triplets_sampling:
            triplet_loss = self.batch_hard_triplet_loss(y_true, pairwise_dist)
        else:
            triplet_loss = self.batch_all_triplet_loss(y_true, pairwise_dist)
        return triplet_loss

    def batch_all_triplet_loss(self, y_true, pairwise_dist):
        anchor_positive_dist = K.expand_dims(pairwise_dist, 2)
        anchor_negative_dist = K.expand_dims(pairwise_dist, 1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        mask = K.cast(self._get_triplet_mask(y_true, pairwise_dist), K.dtype(triplet_loss))
        triplet_loss = mask * triplet_loss
        triplet_loss = K.clip(triplet_loss, 0.0, None)
        valid_triplets = K.cast(K.greater(triplet_loss, 1e-16), K.dtype(triplet_loss))
        num_positive_triplets = K.sum(valid_triplets)
        triplet_loss = K.sum(triplet_loss) / (num_positive_triplets + 1e-16)
        return triplet_loss

    def _get_triplet_mask(self, y_true, pairwise_dist):
        # mask label(a) = label(p)
        mask1 = K.expand_dims(K.equal(K.expand_dims(y_true, 0), K.expand_dims(y_true, 1)), 2)
        mask1 = K.cast(mask1, K.dtype(pairwise_dist))
        # mask a != p
        mask2 = K.expand_dims(K.not_equal(K.expand_dims(pairwise_dist, 2), 0.0), 2)
        mask2 = K.cast(mask2, K.dtype(pairwise_dist))
        # mask label(n) != label(a)
        mask3 = K.expand_dims(K.not_equal(K.expand_dims(y_true, 0), K.expand_dims(y_true, 1)), 1)
        mask3 = K.cast(mask3, K.dtype(pairwise_dist))
        return mask1 * mask2 * mask3

    def train_on_batch(self, batch, y):
        # b = [x for el in batch for x in el]
        if self.token_embeddings and not self.char_embeddings:
            # self.obj_model.train_on_batch(x=[np.asarray(x) for x in batch], y=np.asarray(y))
            loss = self.obj_model.train_on_batch(x=list(batch), y=np.asarray(y))
        elif not self.token_embeddings and self.char_embeddings:
            loss = self.obj_model.train_on_batch(x=[np.asarray(x) for x in batch], y=np.asarray(y))
        elif self.token_embeddings and self.char_embeddings:
            if self.use_matrix:
                loss = self.obj_model.train_on_batch(x=[np.asarray(x) for x in batch], y=np.asarray(y))
            else:
                b = [x[0] for x in batch]
                loss = self.obj_model.train_on_batch(x=b, y=np.asarray(y))
        return loss

    def predict_score_on_batch(self, batch):
        if self.token_embeddings and not self.char_embeddings:
            return self.score_model.predict_on_batch(x=batch)
        elif not self.token_embeddings and self.char_embeddings:
            return self.score_model.predict_on_batch(x=batch)
        elif self.token_embeddings and self.char_embeddings:
            if self.use_matrix:
                return self.score_model.predict_on_batch(x=batch)
            else:
                b = [batch[i][:,:,0] for i in range(len(batch))]
                b = [np.concatenate([b[i], batch[i][:,:,1:]], axis=2) for i in range(len(batch))]
                return self.score_model.predict_on_batch(x=b)

    def predict_embedding_on_batch(self, batch, type='context'):
        if type == 'context':
            embedding = self.context_embedding
        elif type == 'response':
            embedding = self.response_embedding
        if self.token_embeddings and not self.char_embeddings:
            return embedding.predict_on_batch(x=batch)
        elif not self.token_embeddings and self.char_embeddings:
            return embedding.predict_on_batch(x=batch)
        elif self.token_embeddings and self.char_embeddings:
            if self.use_matrix:
                return embedding.predict_on_batch(x=batch)
            else:
                b = [batch[i][:,:,0] for i in range(len(batch))]
                b = [np.concatenate([b[i], batch[i][:,:,1:]], axis=2) for i in range(len(batch))]
                return embedding.predict_on_batch(x=b)

