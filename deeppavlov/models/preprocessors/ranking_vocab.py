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
        num_ranking_samples_train: The number of condidates to perform ranking.
        num_ranking_samples_test: The number of condidates to perform ranking.
        pos_pool_sample: Whether to sample response from `pos_pool` each time when the batch is generated.
            If ``False``, the response from `response` will be used.
        pos_pool_rank: Whether to count samples from the whole `pos_pool` as correct answers in test / validation mode.
        tokenizer: The method to tokenize contexts and responses.
        seed: Random seed.
        hard_triplets_sampling: Whether to use hard triplets sampling to train the model
            i.e. to choose negative samples close to positive ones.
        hardest_positives: Whether to use only one hardest positive sample per each anchor sample.
            It is only used when ``hard_triplets_sampling`` is set to ``True``.
        semi_hard_negatives: Whether hard negative samples should be further away from anchor samples
            than positive samples or not. It is only used when ``hard_triplets_sampling`` is set to ``True``.
        num_hardest_negatives: It is only used when ``hard_triplets_sampling`` is set to ``True``
            and ``semi_hard_negatives`` is set to ``False``.
        embedder: The method providing embeddings for tokens.
        embedding_dim: Dimensionality of token (word) embeddings.
        use_matrix: Whether to use trainable matrix with token (word) embeddings.
        triplet_mode: Whether to use a model with triplet loss.
            If ``False``, a model with crossentropy loss will be used.
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
                 hard_triplets_sampling: bool = False,
                 hardest_positives: bool = False,
                 semi_hard_negatives: bool = False,
                 num_hardest_negatives: int = None,
                 embedder: Callable = "random",
                 embedding_dim: int = 300,
                 triplet_mode: bool = True,
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
        self.upd_embs = update_embeddings
        self.num_ranking_samples = num_ranking_samples
        self.pos_pool_sample = pos_pool_sample
        self.pos_pool_rank = pos_pool_rank
        self.tokenizer = tokenizer
        self.hard_triplets_sampling = hard_triplets_sampling
        self.hardest_positives = hardest_positives
        self.semi_hard_negatives = semi_hard_negatives
        self.num_hardest_negatives = num_hardest_negatives
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self.use_matrix = use_matrix
        self.triplet_mode = triplet_mode

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
        self.response2toks_vocab = {}
        self.response2emb_vocab = {}
        self.context2toks_vocab = {}
        self.context2emb_vocab = {}

        random.seed(seed)

        super().__init__(load_path=self.load_path, save_path=self.save_path, **kwargs)

        if self.embedder == "random":
            self.embeddings_model = dict()

        self.len_vocab = 0
        self.len_char_vocab = 0
        self.emb_matrix = None

        if is_done(self.load_path):
            self.load()

    def fit(self, context, response, pos_pool, neg_pool):
        if not is_done(self.save_path):
            log.info("[initializing new `{}`]".format(self.__class__.__name__))
            if self.char_embeddings:
                self.build_int2char_vocab()
                self.build_char2int_vocab()
            c_tok = self.tokenizer(context)
            r_tok = self.tokenizer(response)
            pos_pool_tok = [self.tokenizer(el) for el in pos_pool]
            if neg_pool[0] is not None:
                neg_pool_tok = [self.tokenizer(el) for el in neg_pool]
            else:
                neg_pool_tok = neg_pool
            self.build_int2tok_vocab(c_tok, r_tok, pos_pool_tok, neg_pool_tok)
            self.build_tok2int_vocab()

            self.len_vocab = len(self.tok2int_vocab)
            self.len_char_vocab = len(self.char2int_vocab)

            self.build_context2toks_vocab(c_tok)
            self.build_response2toks_vocab(r_tok, pos_pool_tok, neg_pool_tok)
            if self.upd_embs:
                self.build_context2emb_vocab()
                self.build_response2emb_vocab()
            self.build_emb_matrix()

    def load(self):
        log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
        if self.char_embeddings:
            self.load_int2char()
            self.build_char2int_vocab()
        self.load_int2tok()
        self.build_tok2int_vocab()
        self.load_context2toks()
        self.load_response2toks()
        if self.upd_embs:
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
        if self.upd_embs:
            self.save_cont()
            self.save_resp()
        mark_done(self.save_path)

    def build_int2char_vocab(self):
        pass

    def build_int2tok_vocab(self, c_tok, r_tok, pos_pool_tok, neg_pool_tok):
        c = set([x for el in c_tok for x in el])
        r = set([x for el in r_tok for x in el])
        ppool = [x for el in pos_pool_tok for x in el]
        ppool = set([x for el in ppool for x in el])
        r = r | ppool
        if neg_pool_tok[0] is not None:
            npool = [x for el in neg_pool_tok for x in el]
            npool = set([x for el in npool for x in el])
            r = r | npool
        tok = c | r
        self.int2tok_vocab = {el[0]+1:el[1] for el in enumerate(tok)}
        self.int2tok_vocab[0] = '<UNK>'

    def build_response2toks_vocab(self, r_tok, pos_pool_tok, neg_pool_tok):
        r = set([' '.join(el) for el in r_tok])
        ppool = [x for el in pos_pool_tok for x in el]
        ppool = set([' '.join(el) for el in ppool])
        r = r | ppool
        if neg_pool_tok[0] is not None:
            npool = [x for el in neg_pool_tok for x in el]
            npool = set([' '.join(el) for el in npool])
            r = r | npool
        self.response2toks_vocab = {el[0]: el[1].split() for el in enumerate(r)}

    def build_context2toks_vocab(self, contexts):
        c = [' '.join(el) for el in contexts]
        c = set(c)
        self.context2toks_vocab = {el[0]: el[1].split() for el in enumerate(c)}


    def build_char2int_vocab(self):
        self.char2int_vocab = {el[1]: el[0] for el in self.int2char_vocab.items()}

    def build_tok2int_vocab(self):
        self.tok2int_vocab = {el[1]: el[0] for el in self.int2tok_vocab.items()}

    def build_response2emb_vocab(self):
        for i in range(len(self.response2toks_vocab)):
            self.response2emb_vocab[i] = None

    def build_context2emb_vocab(self):
        for i in range(len(self.context2toks_vocab)):
            self.context2emb_vocab[i] = None

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

    def __call__(self, context, response, pos_pool, neg_pool):
        c_tok = self.tokenizer(context)
        r_tok = self.tokenizer(response)
        pos_pool_tok = [self.tokenizer(el) for el in pos_pool]
        if neg_pool[0] is not None:
            neg_pool_tok = [self.tokenizer(el[:self.num_ranking_samples]) for el in neg_pool]
        else:
            neg_pool_tok = neg_pool
        c = [el for el in self.make_ints(c_tok)]
        r = [el for el in self.make_ints(r_tok)]
        ppool = [self.make_ints(el) for el in pos_pool_tok]
        if neg_pool[0] is not None:
            npool =[self.make_ints(el) for el in neg_pool_tok]
        else:
            npool = [self.make_ints(self.generate_items(el)) for el in pos_pool_tok]

        if self.hard_triplets_sampling:
            b = self.make_hard_triplets(x, y, self._net)
            y = np.ones(len(b[0][0]))
        else:
            b = self.make_batch(c, r, ppool, npool)

        return b

    def generate_items(self, pos_pool):
        candidates = []
        for i in range(self.num_ranking_samples):
            candidate = self.response2toks_vocab[random.randint(0, len(self.response2toks_vocab)-1)]
            while candidate in pos_pool:
                candidate = self.response2toks_vocab[random.randint(0, len(self.response2toks_vocab)-1)]
            candidates.append(candidate)
        return candidates

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
        return ints_li

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

    def save_context2toks(self):
        with self.cont_save_path.open('w') as f:
            f.write('\n'.join(['\t'.join([str(el[0]), ' '.join(el[1])]) for el in self.context2toks_vocab.items()]))

    def load_context2toks(self):
        with self.cont_load_path.open('r') as f:
            data = f.readlines()
        self.context2toks_vocab = {int(el.split('\t')[0]): el.split('\t')[1][:-1].split(' ') for el in data}

    def save_response2toks(self):
        with self.resp_save_path.open('w') as f:
            f.write(
                '\n'.join(['\t'.join([str(el[0]), ' '.join(el[1])]) for el in self.response2toks_vocab.items()]))

    def load_response2toks(self):
        with self.resp_load_path.open('r') as f:
            data = f.readlines()
        self.response2toks_vocab = {int(el.split('\t')[0]): el.split('\t')[1][:-1].split(' ') for el in data}

    def save_cont(self):
        context_embeddings = []
        for i in range(len(self.context2emb_vocab)):
            context_embeddings.append(self.context2emb_vocab[i])
        context_embeddings = np.vstack(context_embeddings)
        np.save(self.cemb_save_path, context_embeddings)

    def load_cont(self):
        context_embeddings_arr = np.load(self.cemb_load_path)
        for i in range(context_embeddings_arr.shape[0]):
            self.context2emb_vocab[i] = context_embeddings_arr[i]

    def save_resp(self):
        response_embeddings = []
        for i in range(len(self.response2emb_vocab)):
            response_embeddings.append(self.response2emb_vocab[i])
        response_embeddings = np.vstack(response_embeddings)
        np.save(self.remb_save_path, response_embeddings)

    def load_resp(self):
        response_embeddings_arr = np.load(self.remb_load_path)
        for i in range(response_embeddings_arr.shape[0]):
            self.response2emb_vocab[i] = response_embeddings_arr[i]

    def make_batch(self, cont, resp, pos_pool, neg_pool):
        if self.use_matrix:
            context_emb = cont
        else:
            context_emb = self.get_embs(cont)
        if self.pos_pool_sample:
            response = [random.choice(el) for el in pos_pool]
        else:
            response = resp
        if self.use_matrix:
            response_emb = response
        else:
            response_emb = self.get_embs(response)
        if self.triplet_mode:
            negative_response = [random.choice(el) for el in neg_pool]
            if self.use_matrix:
                negative_response_emb = negative_response
            else:
                negative_response_emb = self.get_embs(negative_response)
            if self.hard_triplets_sampling:
                positives = [random.choices(el, k=self.num_positive_samples) for el in pos_pool]
                if self.use_matrix:
                    positives_emb = positives
                else:
                    positives_emb = [self.get_embs(el) for el in positives]
                x = [[(el[0], x) for x in el[1:]] for el in zip(context_emb, *positives_emb)]
            else:
                # x = [(context, response), (context, negative_response)]
                train = [[(el[0], el[1]), (el[0], el[2])]
                         for el in zip(context_emb, response_emb, negative_response_emb)]
        else:
            # x = [(context, response)]
            train = [[(el[0], el[1])] for el in zip(context_emb, response_emb)]
        rank_pool = []
        for i in range(len(context_emb)):
            if self.pos_pool_rank:
                ppool = list(copy.copy(pos_pool[i]))
                ppool.insert(0, ppool.pop(self.get_index(ppool, response[i])))
                rank_pool.append(ppool + list(neg_pool[i][:-len(ppool)]))
            else:
                rank_pool.append([response[i]] + neg_pool[i])
        assert(len(rank_pool[-1]) == self.num_ranking_samples)
        if self.use_matrix:
            rank_pool_emb = rank_pool
        else:
            rank_pool_emb = [self.get_embs(el) for el in rank_pool]
        test = []
        for i in range(len(context_emb)):
            test.append([(context_emb[i], el) for el in rank_pool_emb[i]])
        x = list(zip(train, test))
        return x

    def get_index(self, a, b):
        for i, el in enumerate(a):
            if np.array_equal(el, b):
                return i

    def make_hard_triplets(self, x, y, net):
        samples = [[s[1] for s in el] for el in x]
        labels = y
        batch_size = len(samples)
        num_samples = len(samples[0])
        samp = [y for el in samples for y in el]
        s = self.dict.make_ints(samp)

        embeddings = net.predict_embedding([s, s], 512, type='context')
        embeddings = embeddings / np.expand_dims(np.linalg.norm(embeddings, axis=1), axis=1)
        dot_product = embeddings @ embeddings.T
        square_norm = np.diag(dot_product)
        distances = np.expand_dims(square_norm, 0) - 2.0 * dot_product + np.expand_dims(square_norm, 1)
        distances = np.maximum(distances, 0.0)
        distances = np.sqrt(distances)

        mask_anchor_negative = np.expand_dims(np.repeat(labels, num_samples), 0)\
                               != np.expand_dims(np.repeat(labels, num_samples), 1)
        mask_anchor_negative = mask_anchor_negative.astype(float)
        max_anchor_negative_dist = np.max(distances, axis=1, keepdims=True)
        anchor_negative_dist = distances + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        if self.num_hardest_negatives is not None:
            hard = np.argsort(anchor_negative_dist, axis=1)[:, :self.num_hardest_negatives]
            ind = np.random.randint(self.num_hardest_negatives, size=batch_size * num_samples)
            hardest_negative_ind = hard[batch_size * num_samples * [True], ind]
        else:
            hardest_negative_ind = np.argmin(anchor_negative_dist, axis=1)

        mask_anchor_positive = np.expand_dims(np.repeat(labels, num_samples), 0) \
                               == np.expand_dims(np.repeat(labels, num_samples), 1)
        mask_anchor_positive = mask_anchor_positive.astype(float)
        anchor_positive_dist = mask_anchor_positive * distances

        c =[]
        rp = []
        rn = []
        hrds = []

        if self.hardest_positives:

            if self.semi_hard_negatives:
                hardest_positive_ind = []
                hardest_negative_ind = []
                for p, n in zip(anchor_positive_dist, anchor_negative_dist):
                    no_samples = True
                    p_li = list(zip(p, np.arange(batch_size * num_samples), batch_size * num_samples * [True]))
                    n_li = list(zip(n, np.arange(batch_size * num_samples), batch_size * num_samples * [False]))
                    pn_li = sorted(p_li + n_li, key=lambda el: el[0])
                    for i, x in enumerate(pn_li):
                        if not x[2]:
                            for y in pn_li[:i][::-1]:
                                if y[2] and y[0] > 0.0:
                                    assert (x[1] != y[1])
                                    hardest_negative_ind.append(x[1])
                                    hardest_positive_ind.append(y[1])
                                    no_samples = False
                                    break
                        if not no_samples:
                            break
                    if no_samples:
                        print("There is no negative examples with distances greater than positive examples distances.")
                        exit(0)
            else:
                if self.num_hardest_negatives is not None:
                    hard = np.argsort(anchor_positive_dist, axis=1)[:, -self.num_hardest_negatives:]
                    ind = np.random.randint(self.num_hardest_negatives, size=batch_size * num_samples)
                    hardest_positive_ind = hard[batch_size * num_samples * [True], ind]
                else:
                    hardest_positive_ind = np.argmax(anchor_positive_dist, axis=1)

            for i in range(batch_size):
                for j in range(num_samples):
                    c.append(s[i*num_samples+j])
                    rp.append(s[hardest_positive_ind[i*num_samples+j]])
                    rn.append(s[hardest_negative_ind[i*num_samples+j]])

        else:
            if self.semi_hard_negatives:
                for i in range(batch_size):
                    for j in range(num_samples):
                        for k in range(j+1, num_samples):
                            c.append(s[i*num_samples+j])
                            c.append(s[i*num_samples+k])
                            rp.append(s[i*num_samples+k])
                            rp.append(s[i*num_samples+j])
                            n, hrd = self.get_semi_hard_negative_ind(i, j, k, distances,
                                                                anchor_negative_dist,
                                                                batch_size, num_samples)
                            assert(n != i*num_samples+k)
                            rn.append(s[n])
                            hrds.append(hrd)
                            n, hrd = self.get_semi_hard_negative_ind(i, k, j, distances,
                                                                anchor_negative_dist,
                                                                batch_size, num_samples)
                            assert(n != i*num_samples+j)
                            rn.append(s[n])
                            hrds.append(hrd)
            else:
                for i in range(batch_size):
                    for j in range(num_samples):
                        for k in range(j + 1, num_samples):
                            c.append(s[i * num_samples + j])
                            c.append(s[i * num_samples + k])
                            rp.append(s[i * num_samples + k])
                            rp.append(s[i * num_samples + j])
                            rn.append(s[hardest_negative_ind[i * num_samples + j]])
                            rn.append(s[hardest_negative_ind[i * num_samples + k]])

        triplets = list(zip(c, rp, rn))
        np.random.shuffle(triplets)
        c = [el[0] for el in triplets]
        rp = [el[1] for el in triplets]
        rn = [el[2] for el in triplets]
        ratio = sum(hrds) / len(hrds)
        print("Ratio of semi-hard negative samples is %f" % ratio)
        return [(c, rp), (c, rn)]

    def get_semi_hard_negative_ind(self, i, j, k, distances, anchor_negative_dist, batch_size, num_samples):
        anc_pos_dist = distances[i * num_samples + j, i * num_samples + k]
        neg_dists = anchor_negative_dist[i * num_samples + j]
        n_li_pre = sorted(list(zip(neg_dists, np.arange(batch_size * num_samples))), key=lambda el: el[0])
        n_li = list(filter(lambda x: x[1]<i*num_samples, n_li_pre)) + \
               list(filter(lambda x: x[1]>=(i+1)*num_samples, n_li_pre))
        for x in n_li:
            if x[0] > anc_pos_dist:
                return x[1], True
        return random.choice(n_li)[1], False

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
            emb = np.vstack(emb)
            embs.append(emb)
        # embs = [np.expand_dims(el, axis=0) for el in embs]
        # embs = np.vstack(embs)
        return embs
