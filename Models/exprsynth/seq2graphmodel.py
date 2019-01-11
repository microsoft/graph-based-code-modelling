import random

import numpy as np
from typing import Dict, Any, Tuple, Optional, List

import tensorflow as tf

from .contexttokenmodel import ContextTokenModel, _convert_and_pad_token_sequence
from .model import write_to_minibatch
from .nagdecoder import NAGDecoder


class Seq2GraphModel(ContextTokenModel):
    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = ContextTokenModel.get_default_hyperparameters()
        decoder_defaults = NAGDecoder.get_default_hyperparameters()
        defaults.update(decoder_defaults)
        #TODO: These should be allowed:
        defaults['eg_use_context_attention'] = False
        defaults['eg_use_literal_copying'] = False
        return defaults

    def __init__(self, hyperparameters, run_name: Optional[str]=None, model_save_dir: Optional[str]=None, log_save_dir: Optional[str]=None):
        super().__init__(hyperparameters, run_name, model_save_dir, log_save_dir)
        self._decoder_model = NAGDecoder(self)

    def _make_parameters(self):
        super()._make_parameters()

        cx_hidden_size = self.hyperparameters['hole_rnn_layer_size'] * 2
        eg_hidden_size = self.hyperparameters['eg_hidden_size']
        self.parameters['cx_representation_to_eg_representation'] = \
            tf.get_variable(name='cx_representation_to_eg_representation',
                            initializer=tf.glorot_uniform_initializer(),
                            shape=[cx_hidden_size, eg_hidden_size],
                            )

        self._decoder_model.make_parameters()

    def _make_placeholders(self, is_train: bool) -> None:
        super()._make_placeholders(is_train)

        # Var usage Contexts
        var_usage_context_size = 2 * self.hyperparameters['num_cx_tokens_per_side'] + 1
        max_num_types = self.hyperparameters['cx_max_num_types']

        self.placeholders['cx_var_usage_context_to_var_id'] = tf.placeholder(tf.int32,
                                                                             [None],
                                                                             name='cx_var_usage_context_to_var_id')
        self.placeholders['cx_num_usage_contexts_per_variable'] = tf.placeholder(tf.int32,
                                                                                 [None],
                                                                                 name='cx_num_usage_contexts_per_variable')
        self.placeholders['cx_var_usage_context_token_ids'] = tf.placeholder(tf.int32,
                                                                             shape=[None, var_usage_context_size],
                                                                             name='cx_var_usage_context_token_ids')
        self.placeholders['cx_var_usage_context_type_ids'] = tf.placeholder(tf.int32,
                                                                            shape=[None, var_usage_context_size, max_num_types],
                                                                            name='cx_var_usage_context_type_ids')
        self.placeholders['cx_var_usage_context_type_ids_mask'] = tf.placeholder(tf.bool,
                                                                                 shape=[None, var_usage_context_size, max_num_types],
                                                                                 name='cx_usage_cx_type_ids_mask')

        self.placeholders['eg_node_id_to_cx_repr_id'] = tf.placeholder(tf.int32,
                                                                       [None],
                                                                       name='eg_node_id_to_cx_repr_id')

        self._decoder_model.make_placeholders(is_train)

    def _make_model(self, is_train: bool=True):
        super()._make_model(is_train)

        var_usage_context_embeddings = self._embed_typed_token_sequence(tokens=self.placeholders['cx_var_usage_context_token_ids'],
                                                                        types=self.placeholders['cx_var_usage_context_type_ids'],
                                                                        type_mask=self.placeholders['cx_var_usage_context_type_ids_mask'])

        with tf.name_scope('usage_context_rnn') as scope:
            rnn_size = self.hyperparameters['hole_rnn_layer_size']
            rnn_fw0 = tf.contrib.rnn.GRUCell(rnn_size)
            rnn_bw0 = tf.contrib.rnn.GRUCell(rnn_size)
            rnn_fw1 = tf.contrib.rnn.GRUCell(rnn_size)
            rnn_bw1 = tf.contrib.rnn.GRUCell(rnn_size)

            output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cells_fw=[rnn_fw0, rnn_fw1], cells_bw=[rnn_bw0, rnn_bw1],
                                                                  inputs=tf.unstack(var_usage_context_embeddings, axis=1),
                                                                  dtype=tf.float32, scope=scope)
            var_usage_context_representations = output[self.hyperparameters['num_cx_tokens_per_side']]
            var_usage_representations = tf.segment_sum(data=var_usage_context_representations,
                                                       segment_ids=self.placeholders['cx_var_usage_context_to_var_id'])

            num_usage_contexts_per_variable = tf.to_float(tf.expand_dims(self.placeholders['cx_num_usage_contexts_per_variable'], -1))
            # Add a small number for the cases where a variable has no context tokens
            self.ops['variable_usage_representations'] = \
                var_usage_representations /(num_usage_contexts_per_variable + .00001)  # num_vars x D

        cx_hidden_size = self.hyperparameters['hole_rnn_layer_size'] * 2

        num_cx_tokens_per_side = self.hyperparameters['num_cx_tokens_per_side']
        cx_last_tok_represenstations = self.ops['cx_hole_context_representations'][num_cx_tokens_per_side - 1]
        cx_hole_representations = self.ops['cx_hole_context_representations'][num_cx_tokens_per_side]
        all_context_info_representations = \
            tf.concat([tf.zeros(shape=[1, cx_hidden_size]),
                       self.ops['variable_usage_representations'],
                       cx_hole_representations,
                       cx_last_tok_represenstations],
                      axis=0)
        # Linear layer to uncouple the latent space and dimensionality of encoder and decoder:
        transformed_context_info_representations = tf.matmul(all_context_info_representations,
                                                             self.parameters['cx_representation_to_eg_representation'])
        self.ops['eg_node_representations_from_context'] = tf.gather(params=transformed_context_info_representations,
                                                                     indices=self.placeholders['eg_node_id_to_cx_repr_id'])
        self.ops['eg_node_representation_use_from_context'] = tf.greater(self.placeholders['eg_node_id_to_cx_repr_id'], 0)

        self._decoder_model.make_model(is_train)

    @staticmethod
    def _init_metadata(hyperparameters: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(Seq2GraphModel, Seq2GraphModel)._init_metadata(hyperparameters, raw_metadata)
        NAGDecoder.init_metadata(raw_metadata)

    @staticmethod
    def _load_metadata_from_sample(hyperparameters: Dict[str, Any], raw_sample: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(Seq2GraphModel, Seq2GraphModel)._load_metadata_from_sample(hyperparameters, raw_sample, raw_metadata)
        NAGDecoder.load_metadata_from_sample(raw_sample, raw_metadata)

    def _finalise_metadata(self, raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super()._finalise_metadata(raw_metadata_list)
        self._decoder_model.finalise_metadata(raw_metadata_list, final_metadata)
        return final_metadata

    @staticmethod
    def _load_data_from_sample(hyperparameters: Dict[str, Any],
                               metadata: Dict[str, Any],
                               raw_sample: Dict[str, Any],
                               result_holder: Dict[str, Any],
                               is_train: bool=True) -> bool:
        keep_sample = super(Seq2GraphModel, Seq2GraphModel)._load_data_from_sample(hyperparameters, metadata, raw_sample, result_holder, is_train)
        if not keep_sample:
            return False
        num_cx_tokens_per_side = hyperparameters['num_cx_tokens_per_side']
        max_num_types = hyperparameters['cx_max_num_types']
        full_cx_size = 2 * num_cx_tokens_per_side + 1

        # Variables are sorted alphabetically
        result_holder['cx_sorted_variables_in_scope'] = []
        result_holder['cx_var_usage_context_tokens'] = []
        result_holder['cx_var_usage_context_types'] = []
        result_holder['cx_var_usage_context_types_mask'] = []
        for i, variable_usage_contexts in enumerate(sorted(raw_sample['VariableUsageContexts'], key=lambda cx: cx['Name'])):
            variable_token_idx = metadata['cx_token_vocab'].get_id_or_unk(variable_usage_contexts['Name'])
            variable_node_id = variable_usage_contexts['NodeId']
            variable_type = 'type:' + raw_sample['ContextGraph']['NodeTypes'][str(variable_node_id)]
            variable_type_idxs = metadata['cx_type_vocab'].get_id_or_unk(variable_type)[:max_num_types]

            num_var_usage_contexts = min(len(variable_usage_contexts['TokenContexts']), hyperparameters['max_num_contexts_per_variable'])
            var_usage_context_tokens = np.zeros((num_var_usage_contexts, full_cx_size), dtype=np.int32)
            var_usage_context_types = np.zeros((num_var_usage_contexts, full_cx_size, max_num_types), dtype=np.int32)
            var_usage_context_type_mask = np.zeros((num_var_usage_contexts, full_cx_size, max_num_types), dtype=np.bool)
            assert len(variable_usage_contexts['TokenContexts']) > 0

            random.shuffle(variable_usage_contexts['TokenContexts'])
            for context_idx, usage_context in enumerate(variable_usage_contexts['TokenContexts'][:num_var_usage_contexts]):
                before_context, after_context = usage_context
                before_tokens, before_token_types, before_token_type_masks = \
                    _convert_and_pad_token_sequence(hyperparameters, metadata, before_context, num_cx_tokens_per_side, start_from_left=False)
                after_tokens, after_token_types, after_token_type_masks = \
                    _convert_and_pad_token_sequence(hyperparameters, metadata, after_context, num_cx_tokens_per_side)

                var_usage_context_tokens[context_idx, :num_cx_tokens_per_side] = before_tokens
                var_usage_context_types[context_idx, :num_cx_tokens_per_side] = before_token_types
                var_usage_context_type_mask[context_idx, :num_cx_tokens_per_side] = before_token_type_masks

                var_usage_context_tokens[context_idx][num_cx_tokens_per_side] = variable_token_idx
                var_usage_context_types[context_idx, num_cx_tokens_per_side, :len(variable_type_idxs)] = variable_type_idxs
                var_usage_context_type_mask[context_idx, num_cx_tokens_per_side, :len(variable_type_idxs)] = True

                var_usage_context_tokens[context_idx, num_cx_tokens_per_side + 1:] = after_tokens
                var_usage_context_types[context_idx, num_cx_tokens_per_side + 1:] = after_token_types
                var_usage_context_type_mask[context_idx, num_cx_tokens_per_side + 1:] = after_token_type_masks
                context_idx += 1

            result_holder['cx_sorted_variables_in_scope'].append(variable_usage_contexts['Name'])
            result_holder['cx_var_usage_context_tokens'].append(var_usage_context_tokens)
            result_holder['cx_var_usage_context_types'].append(var_usage_context_types)
            result_holder['cx_var_usage_context_types_mask'].append(var_usage_context_type_mask)

        return NAGDecoder.load_data_from_sample(hyperparameters, metadata, raw_sample, result_holder, is_train)


    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super()._init_minibatch(batch_data)
        batch_data['cx_var_usage_context_tokens'] = []
        batch_data['cx_var_usage_context_types'] = []
        batch_data['cx_var_usage_context_types_mask'] = []
        batch_data['cx_var_usage_context_to_var_id'] = []
        batch_data['cx_num_usage_contexts_per_variable'] = []
        batch_data['cx_var_id_offset'] = 0

        batch_data['eg_node_id_to_cx_repr_id'] = []
        batch_data['cx_var_repr_offset'] = 0
        self._decoder_model.init_minibatch(batch_data)

    @staticmethod
    def __extend_minibatch_with_context_info(batch_data: Dict[str, Any], sample: Dict[str, Any]) -> None:
        batch_data['cx_var_usage_context_tokens'].extend(np.concatenate(sample['cx_var_usage_context_tokens'], axis=0))
        batch_data['cx_var_usage_context_types'].extend(np.concatenate(sample['cx_var_usage_context_types'], axis=0))
        batch_data['cx_var_usage_context_types_mask'].extend(np.concatenate(sample['cx_var_usage_context_types_mask'], axis=0))

        batch_data['eg_initial_node_ids'].append(batch_data['eg_node_offset'])  # hole node is initial
        batch_data['eg_initial_node_ids'].append(batch_data['eg_node_offset'] + 1)  # last token node is initial
        for (var_idx, _) in enumerate(sample['cx_sorted_variables_in_scope']):
            batch_var_idx = batch_data['cx_var_id_offset'] + var_idx
            # We need to map all the var usage contexts to this var id, so duplicate the var id appropriately often:
            num_usage_context_for_var = sample['cx_var_usage_context_tokens'][var_idx].shape[0]
            batch_data['cx_var_usage_context_to_var_id'].extend([batch_var_idx] * num_usage_context_for_var)
            batch_data['cx_num_usage_contexts_per_variable'].append(num_usage_context_for_var)
            batch_data['eg_initial_node_ids'].append(batch_data['eg_node_offset'] + 2 + var_idx)  # last variable use node is initial

        batch_data['cx_var_id_offset'] += len(sample['cx_sorted_variables_in_scope'])

        # To initialise some of the expansion graph nodes, we will need to map these nodes into
        # a big tensor of representations (shape [None, repr_size]) obtained from the context, which will look like this:
        #   [0-element for unassigned nodes,
        #    sample_1_usage_1_repr,
        #    ...
        #    sample_1_usage_K_repr,
        #    sample_2_usage_1_repr,
        #    ...
        #    sample_1_hole_repr,
        #    sample_2_hole_repr,
        #    ...
        #    sample_1_last_tok_repr,
        #    sample_2_last_tok_repr,
        #    ...])
        num_eg_nodes = len(sample['eg_node_labels'])
        this_sample_eg_node_to_repr_index = np.zeros(shape=[num_eg_nodes], dtype=np.int32)
        for var_idx in range(len(sample['cx_sorted_variables_in_scope'])):
            # the first two eg nodes are the hole/root node and the last token, and then the variables start:
            this_sample_eg_node_to_repr_index[2 + var_idx] = 1 + batch_data['cx_var_repr_offset'] + var_idx
        batch_data['eg_node_id_to_cx_repr_id'].append(this_sample_eg_node_to_repr_index)
        batch_data['cx_var_repr_offset'] += len(sample['cx_sorted_variables_in_scope'])

    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        batch_finished = super()._extend_minibatch_by_sample(batch_data, sample)
        self.__extend_minibatch_with_context_info(batch_data, sample)
        self._decoder_model.extend_minibatch_by_sample(batch_data, sample)
        return batch_finished

    def __finalise_minibatch_with_context_info(self, batch_data: Dict[str, Any], final_batch_data: Dict[tf.Tensor, Any]):
        write_to_minibatch(final_batch_data, self.placeholders['cx_var_usage_context_token_ids'], batch_data['cx_var_usage_context_tokens'])
        write_to_minibatch(final_batch_data, self.placeholders['cx_var_usage_context_type_ids'], batch_data['cx_var_usage_context_types'])
        write_to_minibatch(final_batch_data, self.placeholders['cx_var_usage_context_type_ids_mask'], batch_data['cx_var_usage_context_types_mask'])

        write_to_minibatch(final_batch_data, self.placeholders['cx_var_usage_context_to_var_id'], batch_data['cx_var_usage_context_to_var_id'])
        write_to_minibatch(final_batch_data, self.placeholders['cx_num_usage_contexts_per_variable'], batch_data['cx_num_usage_contexts_per_variable'])

        number_of_samples = len(batch_data['eg_node_id_to_cx_repr_id'])
        # hole representations in final representation tensor start at index 1 + batch_data['cx_var_repr_offset'],
        # last token representations  in final representation tensor start at index 1 + batch_data['cx_var_repr_offset'] + number_of_samples
        for (sample_idx, this_sample_eg_node_to_repr_index) in enumerate(batch_data['eg_node_id_to_cx_repr_id']):
            this_sample_eg_node_to_repr_index[0] = 1 + batch_data['cx_var_repr_offset'] + sample_idx  # eg node for hole
            this_sample_eg_node_to_repr_index[1] = 1 + batch_data['cx_var_repr_offset'] + number_of_samples + sample_idx  # eg node for last token
        batch_data['eg_node_id_to_cx_repr_id'] = np.concatenate(batch_data['eg_node_id_to_cx_repr_id'], axis=0)

        write_to_minibatch(final_batch_data, self.placeholders['eg_node_id_to_cx_repr_id'], batch_data['eg_node_id_to_cx_repr_id'])

    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool) -> Dict[tf.Tensor, Any]:
        final_batch_data = super()._finalise_minibatch(batch_data, is_train)
        self.__finalise_minibatch_with_context_info(batch_data, final_batch_data)
        self._decoder_model.finalise_minibatch(batch_data, final_batch_data)
        return final_batch_data

    # ------- These are the bits that we only need for test-time:
    def _tensorise_one_test_sample(self, loaded_sample: Dict[str, Any]) -> Dict[tf.Tensor, Any]:
        test_minibatch = {}
        self._init_minibatch(test_minibatch)

        # Note that we are primarily interested in the node_id_to_cg_node_id (and some labels) for the expansion graph:
        super()._extend_minibatch_by_sample(test_minibatch, loaded_sample)
        self.__extend_minibatch_with_context_info(test_minibatch, loaded_sample)
        final_test_batch = super()._finalise_minibatch(test_minibatch, is_train=False)
        write_to_minibatch(final_test_batch, self.placeholders['eg_node_token_ids'], test_minibatch['eg_node_token_ids'])
        self.__finalise_minibatch_with_context_info(test_minibatch, final_test_batch)

        return final_test_batch

    def _encode_one_test_sample(self, sample_data_dict: Dict[tf.Tensor, Any]) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        return (self.sess.run(self.ops['eg_node_representations_from_context'],
                              feed_dict=sample_data_dict),
                None)