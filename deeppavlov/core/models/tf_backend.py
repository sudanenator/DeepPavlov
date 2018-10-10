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

from abc import ABCMeta
from functools import wraps

from six import with_metaclass
import tensorflow as tf


def _graph_wrap(func, graph):
    """Constructs function encapsulated in the graph."""
    @wraps(func)
    def _wrapped(*args, **kwargs):
        with graph.as_default():
            return func(*args, **kwargs)
    return _wrapped


def _keras_wrap(func, graph, session):
    """Constructs function encapsulated in the graph and the session."""
    import keras.backend as K

    @wraps(func)
    def _wrapped(*args, **kwargs):
        with graph.as_default():
            K.set_session(session)
            return func(*args, **kwargs)
    return _wrapped


def _is_keras_model(cls):
    # may be, there exists a way to avoid this ugly import?
    from .keras_model import KerasModel
    return issubclass(cls, KerasModel)


class TfModelMeta(with_metaclass(type, ABCMeta)):
    """Metaclass that helps all child classes to have their own graph and session."""
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls)
        if _is_keras_model(cls):
            import keras.backend as K
            if K.backend() != 'tensorflow':
                obj.__init__(*args, **kwargs)
                return obj

            K.clear_session()
            obj.graph = tf.Graph()
            with obj.graph.as_default():
                if hasattr(cls, '_config_session'):
                    obj.sess = cls._config_session()
                else:
                    obj.sess = tf.Session()
        else:
            obj.graph = tf.Graph()

        for meth in dir(obj):
            if meth == '__class__':
                continue
            attr = getattr(obj, meth)
            if callable(attr):
                if _is_keras_model(cls):
                    wrapped_attr = _keras_wrap(attr, obj.graph, obj.sess)
                else:
                    wrapped_attr = _graph_wrap(attr, obj.graph)
                setattr(obj, meth, wrapped_attr)
        obj.__init__(*args, **kwargs)
        return obj
