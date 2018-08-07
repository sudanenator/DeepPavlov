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

import nltk
from typing import List

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register("context_response_tokenizer")
class ContextResponseTokenizer(Component):
    """Class for extraction of contexts from special ranking data format."""
    def __init__(*args, **kwargs):
        pass

    def __call__(self, batch):
        """Tokenize given batch

        Args:
            batch: list of text samples

        Returns:
            list of lists of tokens
        """
        resp = [el.split('\t') for el in batch]
        return resp