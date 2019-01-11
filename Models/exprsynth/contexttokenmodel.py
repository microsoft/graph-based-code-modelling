from abc import abstractmethod
from collections import Counter
from itertools import chain
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from dpu_utils.mlutils.vocabulary import Vocabulary
from dpu_utils.codeutils.lattice import LatticeVocabulary

from exprsynth.model import Model, NO_TYPE, write_to_minibatch


def _convert_and_pad_token_sequence(hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                                    token_sequence: List[Tuple[str, str]], vector_size: int, start_from_left: bool = True) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_vocab = metadata['cx_token_vocab']  # type: Vocabulary
    type_vocab = metadata['cx_type_vocab']  # type: LatticeVocabulary

    max_num_types = hyperparameters['cx_max_num_types']

    if start_from_left:
        token_sequence = token_sequence[:vector_size]
    else:
        token_sequence = token_sequence[-vector_size:]

    token_ids = np.zeros(vector_size, dtype=np.int32)
    type_ids = np.zeros((vector_size, max_num_types), dtype=np.int32)
    type_masks = np.zeros((vector_size, max_num_types), dtype=np.bool)
    type_masks[:, 0] = True

    if start_from_left:
        start_idx = 0
    else:
        start_idx = vector_size - len(token_sequence)

    for i, (token, token_type) in enumerate(token_sequence, start=start_idx):
        token_ids[i] = token_vocab.get_id_or_unk(token)

        token_types = type_vocab.get_id_or_unk(token_type)[:max_num_types]
        type_ids[i, :len(token_types)] = token_types
        type_masks[i, :len(token_types)] = True

    return token_ids, type_ids, type_masks


class ContextTokenModel(Model):
    @staticmethod
    @abstractmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = Model.get_default_hyperparameters()
        my_defaults = {
                        "num_cx_tokens_per_side": 10,
                        "cx_token_vocab_size": 5000,
                        "cx_token_representation_size": 128,
                        "cx_max_num_types": 10,
                        "max_num_contexts_per_variable": 5,

                        "cx_type_vocab_size": 5000,
                        "cx_type_representation_size": 32,

                        "hole_rnn_layer_size": 32,
                        "max_num_elements_in_minibatch": 500
                     }
        defaults.update(my_defaults)
        return defaults

    PAD = '%PAD%'

    def __init__(self, hyperparameters, run_name: Optional[str]=None, model_save_dir: Optional[str]=None, log_save_dir: Optional[str]=None):
        super().__init__(hyperparameters, run_name, model_save_dir, log_save_dir)

    @abstractmethod
    def _make_placeholders(self, is_train: bool) -> None:
        super()._make_placeholders(is_train)

        num_cx_tokens_per_side = self.hyperparameters['num_cx_tokens_per_side']
        max_num_types = self.hyperparameters['cx_max_num_types']

        self.placeholders['cx_before_hole_token_ids'] = tf.placeholder(tf.int32,
                                                                       [None, num_cx_tokens_per_side],
                                                                       name='cx_before_hole_token_ids')
        self.placeholders['cx_after_hole_token_ids'] = tf.placeholder(tf.int32,
                                                                      [None, num_cx_tokens_per_side],
                                                                      name='cx_after_hole_token_ids')

        self.placeholders['cx_before_hole_type_ids'] = tf.placeholder(tf.int32,
                                                                      [None, num_cx_tokens_per_side, max_num_types],
                                                                      name='cx_before_hole_type_ids')
        self.placeholders['cx_after_hole_type_ids'] = tf.placeholder(tf.int32,
                                                                     [None, num_cx_tokens_per_side, max_num_types],
                                                                     name='cx_after_hole_type_ids')

        self.placeholders['cx_before_hole_type_ids_mask'] = tf.placeholder(tf.bool,
                                                                           [None, num_cx_tokens_per_side, max_num_types],
                                                                           name='cx_before_hole_type_ids_mask')
        self.placeholders['cx_after_hole_type_ids_mask'] = tf.placeholder(tf.bool,
                                                                          [None, num_cx_tokens_per_side, max_num_types],
                                                                          name='cx_after_hole_type_ids_mask')

    def _embed_typed_token_sequence(self, tokens, types, type_mask):
        token_embeddings = tf.nn.embedding_lookup(params=self.parameters['token_embeddings'], ids=tokens)
        type_embeddings = tf.nn.embedding_lookup(params=self.parameters['type_embeddings'], ids=types)

        token_embeddings = tf.nn.dropout(token_embeddings, keep_prob=self.hyperparameters['dropout_keep_rate'])
        type_embeddings = tf.nn.dropout(type_embeddings, keep_prob=self.hyperparameters['dropout_keep_rate'])

        type_mask = tf.expand_dims(type_mask, -1)
        type_embeddings += (1 - tf.cast(type_mask, tf.float32)) * -1000
        type_embeddings = tf.reduce_max(type_embeddings, axis=2)  # B x cx x D'
        return tf.concat([token_embeddings, type_embeddings], axis=-1)  # B x cx x D+D'

    @abstractmethod
    def _make_model(self, is_train: bool=True) -> None:
        super()._make_model(is_train)

        self.parameters['token_embeddings'] = \
            tf.Variable((tf.random_normal([len(self.metadata['cx_token_vocab']),
                        self.hyperparameters['cx_token_representation_size']], mean=0.0, stddev=1.0)))
        self.parameters['type_embeddings'] = \
            tf.Variable((tf.random_normal([len(self.metadata['cx_type_vocab']),
                        self.hyperparameters['cx_type_representation_size']], mean=0.0, stddev=1.0)))

        self.parameters['hole_representation'] = tf.Variable(
            (tf.random_normal([1, 1, self.hyperparameters['cx_token_representation_size'] +
                               self.hyperparameters['cx_type_representation_size']], mean=0.0, stddev=1.0)))

        cx_before_holes_embeddings = \
            self._embed_typed_token_sequence(tokens=self.placeholders['cx_before_hole_token_ids'],
                                             types=self.placeholders['cx_before_hole_type_ids'],
                                             type_mask=self.placeholders['cx_before_hole_type_ids_mask'])
        cx_after_holes_embeddings = \
            self._embed_typed_token_sequence(tokens=self.placeholders['cx_after_hole_token_ids'],
                                             types=self.placeholders['cx_after_hole_type_ids'],
                                             type_mask=self.placeholders['cx_after_hole_type_ids_mask'])

        num_contexts = tf.shape(cx_before_holes_embeddings)[0]
        hole_full_embeddings = \
            tf.concat([cx_before_holes_embeddings,
                       tf.tile(self.parameters['hole_representation'], multiples=[num_contexts, 1, 1]),
                       cx_after_holes_embeddings],
                      axis=1)  # B x 2*num_cx_tokens+1 x D

        # Run through BiRNN and get representation
        rnn_size = self.hyperparameters['hole_rnn_layer_size']

        with tf.name_scope('hole_rnn') as scope:
            rnn_fw0 = tf.contrib.rnn.GRUCell(rnn_size)
            rnn_bw0 = tf.contrib.rnn.GRUCell(rnn_size)
            rnn_fw1 = tf.contrib.rnn.GRUCell(rnn_size)
            rnn_bw1 = tf.contrib.rnn.GRUCell(rnn_size)

            output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cells_fw=[rnn_fw0, rnn_fw1], cells_bw=[rnn_bw0, rnn_bw1],
                                                                  inputs=tf.unstack(hole_full_embeddings, axis=1),
                                                                  dtype=tf.float32, scope=scope)
            self.ops['cx_hole_context_representations'] = output

    @staticmethod
    @abstractmethod
    def _init_metadata(hyperparameters: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(ContextTokenModel, ContextTokenModel)._init_metadata(hyperparameters, raw_metadata)
        raw_metadata['cx_token_counter'] = Counter()
        raw_metadata['cx_type_tokens'] = []  # type: List[str]

    @staticmethod
    @abstractmethod
    def _load_metadata_from_sample(hyperparameters: Dict[str, Any], raw_sample: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(ContextTokenModel, ContextTokenModel)._load_metadata_from_sample(hyperparameters, raw_sample, raw_metadata)
        for (token, token_type) in chain(raw_sample['HoleTokensBefore'], raw_sample['HoleTokensAfter']):
            raw_metadata['cx_token_counter'][token] += 1
            if token_type == 'NOTYPE':
                raw_metadata['cx_type_tokens'].append(token_type)
            else:
                raw_metadata['cx_type_tokens'].append('type:' + token_type)

        for variable_usage_contexts in raw_sample['VariableUsageContexts']:
            for usage_context in variable_usage_contexts['TokenContexts']:
                for (token, token_type) in chain(*usage_context):
                    raw_metadata['cx_token_counter'][token] += 1
                    if token_type == 'NOTYPE':
                        raw_metadata['cx_type_tokens'].append(token_type)
                    else:
                        raw_metadata['cx_type_tokens'].append('type:' + token_type)

    @abstractmethod
    def _finalise_metadata(self, raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super()._finalise_metadata(raw_metadata_list)

        # First, merge all needed information:
        merged_token_counter = Counter()
        merged_type_tokens = []
        for raw_metadata in raw_metadata_list:
            merged_token_counter += raw_metadata['cx_token_counter']
            merged_type_tokens.extend(raw_metadata['cx_type_tokens'])

        # Store token, type, and production vocabs:
        final_metadata['cx_token_vocab'] = \
            Vocabulary.create_vocabulary(merged_token_counter,
                                         max_size=self.hyperparameters['cx_token_vocab_size'])
        final_metadata['cx_token_vocab'].add_or_get_id(self.PAD)

        final_metadata['cx_type_vocab'] = \
            LatticeVocabulary.get_vocabulary_for(tokens=merged_type_tokens,
                                                 lattice=final_metadata['type_lattice'],
                                                 max_size=self.hyperparameters['cx_type_vocab_size'] - 1)
        final_metadata['cx_type_vocab'].add_or_get_id(NO_TYPE)
        return final_metadata

    @staticmethod
    def __load_contexttokens_data_from_sample(hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                                              raw_sample: Dict[str, Any], result_holder: Dict[str, Any], is_train: bool=True)\
            -> bool:
        num_cx_tokens_per_side = hyperparameters['num_cx_tokens_per_side']
        result_holder['hole_tokens_before'], result_holder['hole_types_before'], result_holder['hole_types_before_mask'] = \
            _convert_and_pad_token_sequence(hyperparameters, metadata,
                                            raw_sample['HoleTokensBefore'], num_cx_tokens_per_side,
                                            start_from_left=False)
        result_holder['hole_tokens_after'], result_holder['hole_types_after'], result_holder['hole_types_after_mask'] = \
            _convert_and_pad_token_sequence(hyperparameters, metadata,
                                            raw_sample['HoleTokensAfter'], num_cx_tokens_per_side)

        return True

    @staticmethod
    @abstractmethod
    def _load_data_from_sample(hyperparameters: Dict[str, Any],
                               metadata: Dict[str, Any],
                               raw_sample: Dict[str, Any],
                               result_holder: Dict[str, Any],
                               is_train: bool=True) -> bool:
        keep_sample = super(ContextTokenModel, ContextTokenModel)._load_data_from_sample(hyperparameters, metadata, raw_sample, result_holder, is_train)
        return keep_sample and ContextTokenModel.__load_contexttokens_data_from_sample(hyperparameters, metadata,
                                                                                       raw_sample=raw_sample, result_holder=result_holder, is_train=is_train)

    @abstractmethod
    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super()._init_minibatch(batch_data)

        batch_data['cx_hole_tokens_before'] = []
        batch_data['cx_hole_types_before'] = []
        batch_data['cx_hole_types_mask_before'] = []
        batch_data['cx_hole_tokens_after'] = []
        batch_data['cx_hole_types_after'] = []
        batch_data['cx_hole_types_mask_after'] = []

        batch_data['num_samples'] = 0

    @abstractmethod
    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        super()._extend_minibatch_by_sample(batch_data, sample)

        batch_data['cx_hole_tokens_before'].append(sample['hole_tokens_before'])
        batch_data['cx_hole_types_before'].append(sample['hole_types_before'])
        batch_data['cx_hole_types_mask_before'].append(sample['hole_types_before_mask'])

        batch_data['cx_hole_tokens_after'].append(sample['hole_tokens_after'])
        batch_data['cx_hole_types_after'].append(sample['hole_types_after'])
        batch_data['cx_hole_types_mask_after'].append(sample['hole_types_after_mask'])

        batch_data['num_samples'] += 1
        return batch_data['num_samples'] >= self.hyperparameters['max_num_elements_in_minibatch']

    @abstractmethod
    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool) -> Dict[tf.Tensor, Any]:
        minibatch = super()._finalise_minibatch(batch_data, is_train)

        write_to_minibatch(minibatch, self.placeholders['cx_before_hole_token_ids'], batch_data['cx_hole_tokens_before'])
        write_to_minibatch(minibatch, self.placeholders['cx_before_hole_type_ids'], batch_data['cx_hole_types_before'])
        write_to_minibatch(minibatch, self.placeholders['cx_before_hole_type_ids_mask'], batch_data['cx_hole_types_mask_before'])
        write_to_minibatch(minibatch, self.placeholders['cx_after_hole_token_ids'], batch_data['cx_hole_tokens_after'])
        write_to_minibatch(minibatch, self.placeholders['cx_after_hole_type_ids'], batch_data['cx_hole_types_after'])
        write_to_minibatch(minibatch, self.placeholders['cx_after_hole_type_ids_mask'], batch_data['cx_hole_types_mask_after'])

        return minibatch
