import re
from typing import Dict, Any, Tuple, Optional, List

import tensorflow as tf

from .metadata import loader
from .contextgraphmodel import ContextGraphModel
from .model import write_to_minibatch
from .nagdecoder import NAGDecoder


class NAGModel(ContextGraphModel):
    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = ContextGraphModel.get_default_hyperparameters()
        decoder_defaults = NAGDecoder.get_default_hyperparameters()
        defaults.update(decoder_defaults)
        return defaults

    def __init__(self, hyperparameters, run_name: Optional[str]=None, model_save_dir: Optional[str]=None, log_save_dir: Optional[str]=None):
        super().__init__(hyperparameters, run_name, model_save_dir, log_save_dir)
        if hyperparameters['eg_use_literal_copying']:
            assert hyperparameters['eg_hidden_size'] == hyperparameters['cg_ggnn_hidden_size'], \
                "When using copy attention, hidden size of context graph and expansion graph need to be identical"
        self._decoder_model = NAGDecoder(self)

    def _make_parameters(self):
        super()._make_parameters()
        self._decoder_model.make_parameters()

    def _make_placeholders(self, is_train: bool) -> None:
        super()._make_placeholders(is_train)

        # ----- Placeholders for expansion graph:
        # Maps node IDs in the expansion graph to IDs in the context graph; negative value indicates that no
        # corresponding node exists
        self.placeholders['eg_node_id_to_cg_node_id'] = \
            tf.placeholder(dtype=tf.int32,
                           shape=[None],
                           name="eg_node_id_to_cg_node_id")

        if self.hyperparameters['eg_use_literal_copying'] or self.hyperparameters['eg_use_context_attention']:
            self.placeholders['cg_nonkeyword_token_node_ids'] = \
                tf.placeholder(name='cg_nonkeyword_token_node_ids',
                               shape=[None, self.hyperparameters['eg_max_context_tokens']],
                               dtype=tf.int32)  # Shape: [NumGraphs, MaxTokens]
            self.placeholders['context_token_mask'] = \
                tf.placeholder(name='context_token_mask',
                               shape=[None, self.hyperparameters['eg_max_context_tokens']],
                               dtype=tf.float32)  # Shape: [NumGraphs, MaxTokens]

        self._decoder_model.make_placeholders(is_train)

    def _make_model(self, is_train: bool=True):
        super()._make_model(is_train)

        # ----- Compute initial representations for eg nodes, re-using context graph nodes where possible:
        # Add a fresh node ID zero to the representations from the context graph:
        padded_cg_node_representations = tf.concat([tf.zeros(shape=[1, self.hyperparameters['cg_ggnn_hidden_size']]),
                                                    self.ops['cg_node_representations']],
                                                   axis=0)
        # Gather up CG node representations where we have some (otherwise, use the padding):
        self.ops['eg_node_id_to_padded_cg_node_id'] = tf.maximum(self.placeholders['eg_node_id_to_cg_node_id'] + 1, 0)
        eg_node_representations_from_cg = tf.gather(params=padded_cg_node_representations,
                                                    indices=self.ops['eg_node_id_to_padded_cg_node_id'])
        # Linear layer to uncouple the latent space and dimensionality of CG and EG:
        self.ops['eg_node_representations_from_context'] = \
            tf.layers.dense(eg_node_representations_from_cg,
                            units=self.hyperparameters['eg_hidden_size'],
                            use_bias=False,
                            activation=None,
                            kernel_initializer=tf.glorot_uniform_initializer(),
                            )
        self.ops['eg_node_representation_use_from_context'] = tf.greater(self.ops['eg_node_id_to_padded_cg_node_id'], 0)

        if self.hyperparameters["eg_use_literal_copying"] or self.hyperparameters['eg_use_context_attention']:
            cg_nonkeyword_token_node_ids = tf.maximum(self.placeholders['cg_nonkeyword_token_node_ids'] + 1, 0)
            self.ops['context_token_representations'] = tf.gather(params=padded_cg_node_representations,
                                                                  indices=cg_nonkeyword_token_node_ids)

        self._decoder_model.make_model(is_train)

    @staticmethod
    def _init_metadata(hyperparameters: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(NAGModel, NAGModel)._init_metadata(hyperparameters, raw_metadata)
        NAGDecoder.init_metadata(raw_metadata)

    @staticmethod
    def _load_metadata_from_sample(hyperparameters: Dict[str, Any], raw_sample: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(NAGModel, NAGModel)._load_metadata_from_sample(hyperparameters, raw_sample, raw_metadata)
        NAGDecoder.load_metadata_from_sample(raw_sample, raw_metadata)

    def _finalise_metadata(self, raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super()._finalise_metadata(raw_metadata_list)
        self._decoder_model.finalise_metadata(raw_metadata_list, final_metadata)
        final_metadata['nag_reserved_names'] = loader.get_csharp_reserved_names()
        final_metadata['nag_reserved_names'].add("<HOLE>")
        return final_metadata

    @staticmethod
    def _load_data_from_sample(hyperparameters: Dict[str, Any],
                               metadata: Dict[str, Any],
                               raw_sample: Dict[str, Any],
                               result_holder: Dict[str, Any],
                               is_train: bool=True) -> bool:
        keep_sample = super(NAGModel, NAGModel)._load_data_from_sample(hyperparameters, metadata, raw_sample, result_holder, is_train)

        if keep_sample:
            result_holder['eg_node_id_to_cg_node_id'] = {}
            result_holder['eg_node_id_to_cg_node_id'][0] = raw_sample['HoleNode']
            result_holder['eg_node_id_to_cg_node_id'][1] = raw_sample['LastTokenBeforeHole']
            for var_id, (var, cg_graph_var_node_id) in enumerate(raw_sample['LastUseOfVariablesInScope'].items()):
                result_holder['eg_node_id_to_cg_node_id'][2 + var_id] = cg_graph_var_node_id

            if hyperparameters.get("eg_use_literal_copying") or hyperparameters.get("eg_use_context_attention") or False:
                non_keyword_or_punctuation_token_node_ids = []
                non_keyword_or_punctuation_tokens = []
                for (node_id, node_label) in raw_sample['ContextGraph']['NodeLabels'].items():
                    if node_label in metadata['nag_reserved_names']:
                        continue
                    if not re.search('[a-zA-Z0-9]', node_label):
                        continue
                    non_keyword_or_punctuation_token_node_ids.append(int(node_id))
                    non_keyword_or_punctuation_tokens.append(node_label)
                result_holder['cg_nonkeyword_token_node_ids'] = non_keyword_or_punctuation_token_node_ids[:hyperparameters['eg_max_context_tokens']]
                if hyperparameters.get("eg_use_literal_copying"):
                    result_holder['context_nonkeyword_tokens'] = non_keyword_or_punctuation_tokens[:hyperparameters['eg_max_context_tokens']]

            keep_sample = NAGDecoder.load_data_from_sample(hyperparameters, metadata, raw_sample, result_holder, is_train)

        return keep_sample

    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super()._init_minibatch(batch_data)
        batch_data['eg_node_id_to_cg_node_id'] = []
        batch_data['cg_nonkeyword_token_node_ids'] = []
        batch_data['context_token_mask'] = []
        self._decoder_model.init_minibatch(batch_data)

    def __extend_minibatch_by_nag_bridge_from_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> None:
        num_eg_nodes = len(sample['eg_node_labels'])
        original_cg_node_offset = \
            batch_data['cg_node_offset'] - self._get_number_of_nodes_in_graph(sample)  # The offset has already been updated in our superclass

        for eg_node_id in range(num_eg_nodes):
            cg_node_id = sample['eg_node_id_to_cg_node_id'].get(eg_node_id)
            if cg_node_id is None:
                batch_data['eg_node_id_to_cg_node_id'].append(-1)
            else:
                batch_data['eg_node_id_to_cg_node_id'].append(cg_node_id + original_cg_node_offset)
                batch_data['eg_initial_node_ids'].append(eg_node_id + batch_data['eg_node_offset'])

        if self.hyperparameters.get("eg_use_literal_copying") or self.hyperparameters.get("eg_use_context_attention") or False:
            non_keyword_token_node_ids = sample['cg_nonkeyword_token_node_ids']
            # Shift node indices based on other graphs already in batch:
            non_keyword_token_node_ids = [id + original_cg_node_offset for id in non_keyword_token_node_ids]
            # Restrict to right size / pad with -1 to right size and compute mask:
            max_num_tokens = self.hyperparameters['eg_max_context_tokens']
            non_keyword_token_node_ids = non_keyword_token_node_ids[:max_num_tokens]
            num_padding_needed = max_num_tokens - len(non_keyword_token_node_ids)
            non_keyword_token_mask = [1] * len(non_keyword_token_node_ids) + [0] * num_padding_needed
            non_keyword_token_node_ids.extend([-1] * num_padding_needed)
            batch_data['cg_nonkeyword_token_node_ids'].append(non_keyword_token_node_ids)
            batch_data['context_token_mask'].append(non_keyword_token_mask)

    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        batch_finished = super()._extend_minibatch_by_sample(batch_data, sample)
        self.__extend_minibatch_by_nag_bridge_from_sample(batch_data, sample)
        self._decoder_model.extend_minibatch_by_sample(batch_data, sample)

        return batch_finished

    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool) -> Dict[tf.Tensor, Any]:
        minibatch = super()._finalise_minibatch(batch_data, is_train)
        self._decoder_model.finalise_minibatch(batch_data, minibatch)
        write_to_minibatch(minibatch, self.placeholders['eg_node_id_to_cg_node_id'], batch_data['eg_node_id_to_cg_node_id'])
        if self.hyperparameters["eg_use_literal_copying"] or self.hyperparameters['eg_use_context_attention']:
            write_to_minibatch(minibatch, self.placeholders['cg_nonkeyword_token_node_ids'], batch_data['cg_nonkeyword_token_node_ids'])
            write_to_minibatch(minibatch, self.placeholders['context_token_mask'], batch_data['context_token_mask'])
        return minibatch

    # ------- These are the bits that we only need for test-time:
    def _tensorise_one_test_sample(self, loaded_sample: Dict[str, Any]) -> Dict[tf.Tensor, Any]:
        test_minibatch = {}
        self._init_minibatch(test_minibatch)

        # Note that we are primarily interested in the node_id_to_cg_node_id (and some labels) for the expansion graph:
        super()._extend_minibatch_by_sample(test_minibatch, loaded_sample)
        self.__extend_minibatch_by_nag_bridge_from_sample(test_minibatch, loaded_sample)
        final_test_minibatch = super()._finalise_minibatch(test_minibatch, is_train=False)
        write_to_minibatch(final_test_minibatch, self.placeholders['eg_node_token_ids'], test_minibatch['eg_node_token_ids'])
        write_to_minibatch(final_test_minibatch, self.placeholders['eg_node_id_to_cg_node_id'], test_minibatch['eg_node_id_to_cg_node_id'])
        if self.hyperparameters["eg_use_literal_copying"] or self.hyperparameters['eg_use_context_attention']:
            write_to_minibatch(final_test_minibatch, self.placeholders['cg_nonkeyword_token_node_ids'], test_minibatch['cg_nonkeyword_token_node_ids'])
            write_to_minibatch(final_test_minibatch, self.placeholders['context_token_mask'], test_minibatch['context_token_mask'])

        return final_test_minibatch

    def _encode_one_test_sample(self, sample_data_dict: Dict[tf.Tensor, Any]) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        if self.hyperparameters["eg_use_literal_copying"] or self.hyperparameters['eg_use_context_attention']:
            return tuple(self.sess.run([self.ops['eg_node_representations_from_context'],
                                        self.ops['context_token_representations']],
                                       feed_dict=sample_data_dict))
        else:
            return (self.sess.run(self.ops['eg_node_representations_from_context'],
                                  feed_dict=sample_data_dict),
                    None)
