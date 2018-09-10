from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

import numpy as np
import random
from typing import Dict, List, Tuple


@register('siamese_iterator')
class SiameseIterator(DataLearningIterator):
    """The class contains methods for iterating over a dataset for ranking in training, validation and test mode.

    Note:
        Each sample in ``data['train']`` is arranged as follows:
        ``{'context': 21507, 'response': 7009, 'pos_pool': [7009, 7010], 'neg_pool': None}``.
        The context has a 'context' key in the data sample.
        It is represented by a single integer.
        The correct response has the 'response' key in the sample,
        its value is  also always a single integer.
        The list of possible correct responses (there may be several) can be
        obtained
        with the 'pos\_pool' key.
        The value of the 'response' should be equal to the one item from the
        list
        obtained using the 'pos\_pool' key.
        The list of possible negative responses (there can be a lot of them,
        100–10000) is represented by the key 'neg\_pool'.
        Its value is None, when global sampling is used, or the list of fixed
        length, when sampling from predefined negative responses is used.
        It is important that values in 'pos\_pool' and 'negative\_pool' do
        not overlap.
        Single items in 'context', 'response', 'pos\_pool', 'neg\_pool' are
        represented
        by single integers that give lists of integers
        using some dictionary `integer–list of integers`.
        These lists of integers are converted to lists of tokens with
        some dictionary `integer–token`.
        Samples in ``data['valid']`` and ``data['test']`` representation are almost the same
        as the train sample shown above.

    Args:
        data: A dictionary containing training, validation and test parts of the dataset obtainable via
            ``train``, ``valid`` and ``test`` keys.
        seed: Random seed.
        shuffle: Whether to shuffle data.
        random_batches: Whether to choose batches randomly or iterate over data sequentally in training mode.
        batches_per_epoch: A number of batches to choose per each epoch in training mode.
            Only required if ``random_batches`` is set to ``True``.
    """

    def __init__(self,
                 data: Dict[str, List],
                 seed: int = None,
                 shuffle: bool = False,
                 random_batches: bool = False,
                 batches_per_epoch: int = None):

        self.random_batches = random_batches
        self.batches_per_epoch = batches_per_epoch

        np.random.seed(seed)
        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

        super().__init__(self.data, seed=seed, shuffle=shuffle)


    def gen_batches(self, batch_size: int, data_type: str = "train", shuffle: bool = True)->\
            Tuple[List[List[Tuple[int, int]]], List[int]]:
        """Generate batches of inputs and expected outputs to train neural networks.

        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching

        Returns:
            A tuple of a batch of inputs and a batch of expected outputs.

            Inputs and expected outputs have different structure and meaning
            depending on class attributes values and ``data_type``.
        """
        data = self.data[data_type]
        if self.random_batches and self.batches_per_epoch is not None and data_type == "train":
            num_steps = self.batches_per_epoch
            assert(batch_size <= len(data))
        else:
            num_steps = len(data) // batch_size
        if data_type == "train":
            if shuffle:
                np.random.shuffle(data)
            for i in range(num_steps):
                if self.random_batches:
                    context_response_data = np.random.choice(data, size=batch_size, replace=False)
                else:
                    context_response_data = data[i * batch_size:(i + 1) * batch_size]
                yield tuple(zip(*context_response_data))
        if data_type in ["valid", "test"]:
            for i in range(num_steps + 1):
                if i < num_steps:
                    context_response_data = data[i * batch_size:(i + 1) * batch_size]
                else:
                    if len(data[i * batch_size:len(data)]) > 0:
                        context_response_data = data[i * batch_size:len(data)]
                yield tuple(zip(*context_response_data))
