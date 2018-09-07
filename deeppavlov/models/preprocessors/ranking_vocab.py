# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from deeppavlov.core.commands.utils import expand_path
from keras.preprocessing.sequence import pad_sequences
from deeppavlov.core.common.log import get_logger
import random
import copy

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator
from typing import List, Callable
from deeppavlov.core.data.utils import mark_done, is_done

log = get_logger(__name__)


@register('ranking_vocab')
class RankingVocab(Estimator):
    """Class to encode characters, tokens, whole contexts and responses with vocabularies, to pad and truncate.

    Args:
        max_sequence_length: A maximum length of a sequence in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        max_token_length: A maximum length of a token for representing it by a character-level embedding.
        padding: Padding. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a sequence will be padded at the beginning.
            If set to ``post`` it will padded at the end.
        truncating: Truncating. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a sequence will be truncated at the beginning.
            If set to ``post`` it will truncated at the end.
        token_embeddings: Whether to use token embeddins or not.
        char_embeddings: Whether to use character embeddings or not.
        char_pad: Character-level padding. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a token will be padded at the beginning.
            If set to ``post`` it will padded at the end.
        char_trunc: Character-level truncating. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a token will be truncated at the beginning.
            If set to ``post`` it will truncated at the end.
        tok_dynamic_batch:  Whether to use dynamic batching. If ``True``, a maximum length of a sequence for a batch
            will be equal to the maximum of all sequences lengths from this batch,
            but not higher than ``max_sequence_length``.
        char_dynamic_batch: Whether to use dynamic batching for character-level embeddings.
            If ``True``, a maximum length of a token for a batch
            will be equal to the maximum of all tokens lengths from this batch,
            but not higher than ``max_token_length``.
        update_embeddings: Whether to store and update context and response embeddings or not.
        pos_pool_sample: Whether to sample response from `pos_pool` each time when the batch is generated.
            If ``False``, the response from `response` will be used.
        pos_pool_rank: Whether to count samples from the whole `pos_pool` as correct answers in test / validation mode.
        tokenizer: The method to tokenize contexts and responses.
        seed: Random seed.
        embedder: The method providing embeddings for tokens.
        embedding_dim: Dimensionality of token (word) embeddings.
        use_matrix: Whether to use trainable matrix with token (word) embeddings.
    """

    def __init__(self,
                 save_path: str,
                 load_path: str,
                 max_sequence_length: int,
                 use_matrix: bool,
                 max_token_length: int = None,
                 padding: str = 'post',
                 truncating: str = 'post',
                 token_embeddings: bool = True,
                 char_embeddings: bool = False,
                 char_pad: str = 'post',
                 char_trunc: str = 'post',
                 tok_dynamic_batch: bool = False,
                 char_dynamic_batch: bool = False,
                 update_embeddings: bool = False,
                 num_ranking_samples: int = 10,
                 pos_pool_sample: bool = False,
                 pos_pool_rank: bool = True,
                 tokenizer: Callable = None,
                 seed: int = None,
                 embedder: Callable = "random",
                 embedding_dim: int = 300,
                 **kwargs):

        self.max_sequence_length = max_sequence_length
        self.token_embeddings = token_embeddings
        self.char_embeddings = char_embeddings
        self.max_token_length = max_token_length
        self.padding = padding
        self.truncating = truncating
        self.char_pad = char_pad
        self.char_trunc = char_trunc
        self.tok_dynamic_batch = tok_dynamic_batch
        self.char_dynamic_batch = char_dynamic_batch
        self.update_embeddings = update_embeddings
        self.num_ranking_samples = num_ranking_samples
        self.pos_pool_sample = pos_pool_sample
        self.pos_pool_rank = pos_pool_rank
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self.use_matrix = use_matrix


        self.save_path = expand_path(save_path).resolve()
        self.load_path = expand_path(load_path).resolve()

        self.char_save_path = self.save_path / "char2int.dict"
        self.char_load_path = self.load_path / "char2int.dict"
        self.tok_save_path = self.save_path / "tok2int.dict"
        self.tok_load_path = self.load_path / "tok2int.dict"
        self.cont_save_path = self.save_path / "cont2toks.dict"
        self.cont_load_path = self.load_path / "cont2toks.dict"
        self.resp_save_path = self.save_path / "resp2toks.dict"
        self.resp_load_path = self.load_path / "resp2toks.dict"
        self.cemb_save_path = str(self.save_path / "context_embs.npy")
        self.cemb_load_path = str(self.load_path / "context_embs.npy")
        self.remb_save_path = str(self.save_path / "response_embs.npy")
        self.remb_load_path = str(self.load_path / "response_embs.npy")

        self.char2int_vocab = {}
        self.int2char_vocab = {}
        self.int2tok_vocab = {}
        self.tok2int_vocab = {}
        self.int2context_vocab = {}
        self.context2emb_vocab = {}
        self.int2response_vocab = {}
        self.response2emb_vocab = {}

        random.seed(seed)

        super().__init__(load_path=self.load_path, save_path=self.save_path, **kwargs)

        # if self.embedder == "random":
        #     self.embeddings_model = dict()
        self.embedder = embedder

        self.len_vocab = 0
        self.len_char_vocab = 0
        self.emb_matrix = None

        if is_done(self.load_path):
            self.load()

    def fit(self, x):
        if not is_done(self.save_path):
            log.info("[initializing new `{}`]".format(self.__class__.__name__))
            if self.char_embeddings:
                self.build_int2char_vocab()
                self.build_char2int_vocab()
            x_tok = [self.tokenizer(el) for el in x]
            self.build_int2tok_vocab(x_tok)
            self.build_tok2int_vocab()

            self.len_vocab = len(self.tok2int_vocab)
            self.len_char_vocab = len(self.char2int_vocab)

            self.build_int2context_vocab(x)
            self.build_int2response_vocab(x)
            if self.update_embeddings:
                self.build_context2emb_vocab()
                self.build_response2emb_vocab()
            self.build_emb_matrix()

    def __call__(self, x):
        x_cut = [el[:self.num_ranking_samples+1] for el in x]
        x_tok = [self.tokenizer(el) for el in x_cut]
        x_proc = [self.make_ints(el) for el in x_tok]
        if not self.use_matrix:
            x_proc = self.get_embs(x_proc)
        return x_proc


    def load(self):
        log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
        if self.char_embeddings:
            self.load_int2char()
            self.build_char2int_vocab()
        self.load_int2tok()
        self.build_tok2int_vocab()
        self.load_context2toks()
        self.load_response2toks()
        if self.update_embeddings:
            self.load_cont()
            self.load_resp()

        self.len_vocab = len(self.tok2int_vocab)
        self.len_char_vocab = len(self.char2int_vocab)
        if not self.use_matrix:
            self.build_emb_matrix()

    def save(self):
        log.info("[saving `{}`]".format(self.__class__.__name__))
        if not is_done(self.save_path):
            self.save_path.mkdir()
        if self.char_embeddings:
            self.save_int2char()
        self.save_int2tok()
        self.save_context2toks()
        self.save_response2toks()
        if self.update_embeddings:
            self.save_cont()
            self.save_resp()
        mark_done(self.save_path)

    def build_int2char_vocab(self):
        pass

    def build_int2tok_vocab(self, x):
        x_proc = [li for el in x for li in el]
        x_proc = [tok for el in x_proc for tok in el]
        tok = set(x_proc)
        self.int2tok_vocab = {el[0]+1:el[1] for el in enumerate(tok)}
        self.int2tok_vocab[0] = '<UNK>'

    def build_char2int_vocab(self):
        self.char2int_vocab = {el[1]: el[0] for el in self.int2char_vocab.items()}

    def build_tok2int_vocab(self):
        self.tok2int_vocab = {el[1]: el[0] for el in self.int2tok_vocab.items()}

    def build_int2response_vocab(self, x):
        r = [el[1] for el in x]
        r = set(r)
        self.int2response_vocab = {el[0]: el[1] for el in enumerate(r)}

    def build_int2context_vocab(self, x):
        c = [el[0] for el in x]
        c = set(c)
        self.int2response_vocab = {el[0]: el[1] for el in enumerate(c)}

    def build_response2emb_vocab(self):
        self.response2emb_vocab = {el: None for el in self.int2response_vocab.values()}

    def build_context2emb_vocab(self):
        self.context2emb_vocab = {el: None for el in self.int2context_vocab.values()}

    def conts2toks(self, conts_li):
        toks_li = [self.context2toks_vocab[cont] for cont in conts_li]
        return toks_li

    def resps2toks(self, resps_li):
        toks_li = [self.response2toks_vocab[resp] for resp in resps_li]
        return toks_li

    def make_toks(self, items_li, type):
        if type == "context":
            toks_li = self.conts2toks(items_li)
        elif type == "response":
            toks_li = self.resps2toks(items_li)
        return toks_li

    def make_ints(self, toks_li):
        if self.tok_dynamic_batch:
            msl = min(max([len(el) for el in toks_li]), self.max_sequence_length)
        else:
            msl = self.max_sequence_length
        if self.char_dynamic_batch:
            mtl = min(max(len(x) for el in toks_li for x in el), self.max_token_length)
        else:
            mtl = self.max_token_length

        if self.token_embeddings and not self.char_embeddings:
            return self.make_tok_ints(toks_li, msl)
        elif not self.token_embeddings and self.char_embeddings:
            return self.make_char_ints(toks_li, msl, mtl)
        elif self.token_embeddings and self.char_embeddings:
            tok_ints = self.make_tok_ints(toks_li, msl)
            char_ints = self.make_char_ints(toks_li, msl, mtl)
            return np.concatenate([np.expand_dims(tok_ints, axis=2), char_ints], axis=2)

    def make_tok_ints(self, toks_li, msl):
        ints_li = []
        for toks in toks_li:
            ints = []
            for tok in toks:
                index = self.tok2int_vocab.get(tok)
                if self.tok2int_vocab.get(tok) is not None:
                    ints.append(index)
                else:
                    ints.append(0)
            ints_li.append(ints)
        ints_li = pad_sequences(ints_li,
                                maxlen=msl,
                                padding=self.padding,
                                truncating=self.truncating)
        return list(ints_li)

    def make_char_ints(self, toks_li, msl, mtl):
        ints_li = np.zeros((len(toks_li), msl, mtl))

        for i, toks in enumerate(toks_li):
            if self.truncating == 'post':
                toks = toks[:msl]
            else:
                toks = toks[-msl:]
            for j, tok in enumerate(toks):
                if self.padding == 'post':
                    k = j
                else:
                    k = j + msl - len(toks)
                ints = []
                for char in tok:
                    index = self.char2int_vocab.get(char)
                    if index is not None:
                        ints.append(index)
                    else:
                        ints.append(0)
                if self.char_trunc == 'post':
                    ints = ints[:mtl]
                else:
                    ints = ints[-mtl:]
                if self.char_pad == 'post':
                    ints_li[i, k, :len(ints)] = ints
                else:
                    ints_li[i, k, -len(ints):] = ints
        return ints_li

    def save_int2char(self):
        with self.char_save_path.open('w') as f:
            f.write('\n'.join(['\t'.join([str(el[0]), el[1]]) for el in self.int2char_vocab.items()]))

    def load_int2char(self):
        with self.char_load_path.open('r') as f:
            data = f.readlines()
        self.int2char_vocab = {int(el.split('\t')[0]): el.split('\t')[1][:-1] for el in data}

    def save_int2tok(self):
        with open(self.tok_save_path, 'w') as f:
            f.write('\n'.join(['\t'.join([str(el[0]), el[1]]) for el in self.int2tok_vocab.items()]))

    def load_int2tok(self):
        with self.tok_load_path.open('r') as f:
            data = f.readlines()
        self.int2tok_vocab = {int(el.split('\t')[0]): el.split('\t')[1][:-1] for el in data}

    def save_int2context(self):
        with self.cont_save_path.open('w') as f:
            f.write('\n'.join(['\t'.join([str(el[0]), ' '.join(el[1])]) for el in self.context2toks_vocab.items()]))

    def load_int2context(self):
        with self.cont_load_path.open('r') as f:
            data = f.readlines()
        self.int2context_vocab = {int(el.split('\t')[0]): el.split('\t')[1][:-1] for el in data}

    def save_int2response(self):
        with self.resp_save_path.open('w') as f:
            f.write(
                '\n'.join(['\t'.join([str(el[0]), el[1]]) for el in self.response2toks_vocab.items()]))

    def load_int2response(self):
        with self.resp_load_path.open('r') as f:
            data = f.readlines()
        self.int2response_vocab = {int(el.split('\t')[0]): el.split('\t')[1][:-1] for el in data}

    def save_cont(self):
        context_embeddings = list(self.context2emb_vocab.values())
        context_embeddings = np.vstack(context_embeddings)
        np.save(self.cemb_save_path, context_embeddings)

    def load_cont(self):
        context_embeddings_arr = np.load(self.cemb_load_path)
        for i in range(context_embeddings_arr.shape[0]):
            self.context2emb_vocab[i] = context_embeddings_arr[i]

    def save_resp(self):
        response_embeddings = list(self.response2emb_vocab.values())
        response_embeddings = np.vstack(response_embeddings)
        np.save(self.remb_save_path, response_embeddings)

    def load_resp(self):
        response_embeddings_arr = np.load(self.remb_load_path)
        for i in range(response_embeddings_arr.shape[0]):
            self.response2emb_vocab[i] = response_embeddings_arr[i]

    def build_emb_matrix(self):
        self.emb_matrix = np.zeros((len(self.tok2int_vocab), self.embedding_dim))
        for tok, i in self.tok2int_vocab.items():
            if tok == '<UNK>':
                self.emb_matrix[i] = np.random.uniform(-0.6, 0.6, self.embedding_dim)
            else:
                try:
                    self.emb_matrix[i] = self.embeddings_model[tok]
                except:
                    self.emb_matrix[i] = np.random.uniform(-0.6, 0.6, self.embedding_dim)
        del self.embeddings_model

    def get_embs(self, ints):
        embs = []
        for el in ints:
            emb = []
            for int_tok in el:
                assert type(int_tok) != int
                emb.append(self.emb_matrix[int_tok])
            embs.append(emb)
        return embs
