"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from overrides import overrides
from copy import deepcopy
import inspect
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.ranking.ranking_network import RankingNetwork
from deeppavlov.core.common.log import get_logger
from typing import Union, List, Tuple, Dict

log = get_logger(__name__)


@register('ranking_model')
class RankingModel(NNModel):
    """Class to perform ranking.

    Args:
        interact_pred_num: The number of the most relevant contexts and responses
            which model returns in the `interact` regime.
        triplet_mode: Whether to use a model with triplet loss.
            If ``False``, a model with crossentropy loss will be used.
        update_embeddings: Whether to store and update context and response embeddings or not.
        **kwargs: Other parameters.
    """

    def __init__(self,
                 len_vocab: int,
                 len_char_vocab: int,
                 interact_pred_num: int = 3,
                 triplet_mode: bool = True,
                 update_embeddings: bool = False,
                 **kwargs):

        # Parameters for parent classes
        save_path = kwargs.get('save_path', None)
        load_path = kwargs.get('load_path', None)
        train_now = kwargs.get('train_now', None)
        mode = kwargs.get('mode', None)

        super().__init__(save_path=save_path, load_path=load_path,
                         train_now=train_now, mode=mode)

        self.interact_pred_num = interact_pred_num
        self.train_now = train_now
        self.triplet_mode = triplet_mode
        self.upd_embs = update_embeddings
        self.len_vocab = len_vocab
        self.len_char_vocab = len_char_vocab

        opt = deepcopy(kwargs)

        network_parameter_names = list(inspect.signature(RankingNetwork.__init__).parameters)
        self.network_parameters = {par: opt[par] for par in network_parameter_names if par in opt}

        self.load()

        train_parameters_names = list(inspect.signature(self._net.train_on_batch).parameters)
        self.train_parameters = {par: opt[par] for par in train_parameters_names if par in opt}


    @overrides
    def load(self):
        """Load the model from the last checkpoint if it exists. Otherwise instantiate a new model."""
        self._net = RankingNetwork(chars_num=self.len_char_vocab,
                                   toks_num=self.len_vocab,
                                   **self.network_parameters)

    @overrides
    def save(self):
        """Save the model."""
        log.info('[saving model to {}]'.format(self.save_path.resolve()))
        self._net.save()
        if self.upd_embs:
            self.set_embeddings()
        # self.embdict.save()

    @check_attr_true('train_now')
    def train_on_batch(self, batch, y):
        """Train the model on a batch."""
        if self.upd_embs:
            self.reset_embeddings()
        b  = [el[0] for el in batch]
        b = self.make_batch(b)
        self._net.train_on_batch(b, y)

    def __call__(self, batch):
        """Make a prediction on a batch."""
        if len(batch) > 1:
            y_pred = []
            b = [el[1] for el in batch]
            b = self.make_batch(b)
            for el in b:
                yp = self._net.predict_score_on_batch(el)
                y_pred.append(yp)
            y_pred = np.hstack(y_pred)
            return y_pred

        else:
            c_input = tokenize(batch)
            c_input = self.dict.make_ints(c_input)
            c_input_emb = self._net.predict_embedding_on_batch([c_input, c_input], type='context')

            c_emb = [self.dict.context2emb_vocab[i] for i in range(len(self.dict.context2emb_vocab))]
            c_emb = np.vstack(c_emb)
            pred_cont = np.sum(c_input_emb * c_emb, axis=1)\
                     / np.linalg.norm(c_input_emb, axis=1) / np.linalg.norm(c_emb, axis=1)
            pred_cont = np.flip(np.argsort(pred_cont), 0)[:self.interact_pred_num]
            pred_cont = [' '.join(self.dict.context2toks_vocab[el]) for el in pred_cont]

            r_emb = [self.dict.response2emb_vocab[i] for i in range(len(self.dict.response2emb_vocab))]
            r_emb = np.vstack(r_emb)
            pred_resp = np.sum(c_input_emb * r_emb, axis=1)\
                     / np.linalg.norm(c_input_emb, axis=1) / np.linalg.norm(r_emb, axis=1)
            pred_resp = np.flip(np.argsort(pred_resp), 0)[:self.interact_pred_num]
            pred_resp = [' '.join(self.dict.response2toks_vocab[el]) for el in pred_resp]
            y_pred = [{"contexts": pred_cont, "responses": pred_resp}]
            return y_pred

    def set_embeddings(self):
        if self.dict.response2emb_vocab[0] is None:
            r = []
            for i in range(len(self.dict.response2toks_vocab)):
                r.append(self.dict.response2toks_vocab[i])
            r = self.dict.make_ints(r)
            response_embeddings = self._net.predict_embedding([r, r], 512, type='response')
            for i in range(len(self.dict.response2toks_vocab)):
                self.dict.response2emb_vocab[i] = response_embeddings[i]
        if self.dict.context2emb_vocab[0] is None:
            c = []
            for i in range(len(self.dict.context2toks_vocab)):
                c.append(self.dict.context2toks_vocab[i])
            c = self.dict.make_ints(c)
            context_embeddings = self._net.predict_embedding([c, c], 512, type='context')
            for i in range(len(self.dict.context2toks_vocab)):
                self.dict.context2emb_vocab[i] = context_embeddings[i]

    def reset_embeddings(self):
        if self.dict.response2emb_vocab[0] is not None:
            for i in range(len(self.dict.response2emb_vocab)):
                self.dict.response2emb_vocab[i] = None
        if self.dict.context2emb_vocab[0] is not None:
            for i in range(len(self.dict.context2emb_vocab)):
                self.dict.context2emb_vocab[i] = None

    def shutdown(self):
        pass

    def reset(self):
        pass

    def make_batch(self, x):
        sample_len = len(x[0])
        b = []
        for i in range(sample_len):
            c = []
            r = []
            for el in x:
                c.append(el[i][0])
                r.append(el[i][1])
            b.append([np.asarray(c), np.asarray(r)])
        return b

def tokenize(sen_list):
    sen_tokens_list = []
    for sen in sen_list:
        sent_toks = sent_tokenize(sen)
        word_toks = [word_tokenize(el) for el in sent_toks]
        tokens = [val for sublist in word_toks for val in sublist]
        tokens = [el for el in tokens if el != '']
        sen_tokens_list.append(tokens)
    return sen_tokens_list
