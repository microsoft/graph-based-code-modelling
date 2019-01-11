import numpy as np
from typing import List, Dict, Any, Tuple, Optional

import tensorflow as tf

from .contextgraphmodel import ContextGraphModel
from .metadata import loader
from .model import write_to_minibatch
from .seqdecoder import SeqDecoder

START_TOKEN = '%start%'
END_TOKEN = '%end%'


class Graph2SeqModel(ContextGraphModel):
    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = ContextGraphModel.get_default_hyperparameters()
        decoder_defaults = SeqDecoder.get_default_hyperparameters()
        defaults.update(decoder_defaults)
        return defaults

    def __init__(self, hyperparameters, run_name: Optional[str]=None, model_save_dir: Optional[str]=None, log_save_dir: Optional[str]=None):
        super().__init__(hyperparameters, run_name, model_save_dir, log_save_dir)
        self._decoder_model = SeqDecoder(self)

    def _make_parameters(self):
        super()._make_parameters()

        cg_hidden_size = self.hyperparameters['cg_ggnn_hidden_size']
        self.parameters['cg_representation_to_dec_hidden'] = \
            tf.get_variable(name='cg_representation_to_dec_hidden',
                            initializer=tf.glorot_uniform_initializer(),
                            shape=[cg_hidden_size, self.hyperparameters['decoder_rnn_hidden_size']],
                            )

        self._decoder_model.make_parameters()

    def _make_placeholders(self, is_train: bool) -> None:
        super()._make_placeholders(is_train)
        self.placeholders['root_hole_node_id'] = tf.placeholder(tf.int32,
                                                                shape=(None,),
                                                                name="root_hole_node_id")
        self._decoder_model.make_placeholders(is_train)

    def _make_model(self, is_train: bool=True):
        super()._make_model(is_train)

        # Gather up CG node representations for the hole nodes:
        eg_node_representations_from_cg = tf.gather(params=self.ops['cg_node_representations'],
                                                    indices=self.placeholders['root_hole_node_id'])
        # Linear layer to uncouple the latent space and dimensionality of CG and decoder:
        self.ops['decoder_initial_state'] = tf.matmul(eg_node_representations_from_cg,
                                                      self.parameters['cg_representation_to_dec_hidden'])

        self._decoder_model.make_model(is_train)

    @staticmethod
    def _init_metadata(hyperparameters: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(Graph2SeqModel, Graph2SeqModel)._init_metadata(hyperparameters, raw_metadata)
        SeqDecoder.init_metadata(raw_metadata)

    @staticmethod
    def _load_metadata_from_sample(hyperparameters: Dict[str, Any], raw_sample: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(Graph2SeqModel, Graph2SeqModel)._load_metadata_from_sample(hyperparameters, raw_sample, raw_metadata)
        SeqDecoder.load_metadata_from_sample(raw_sample, raw_metadata)

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
        keep_sample = super(Graph2SeqModel, Graph2SeqModel)._load_data_from_sample(hyperparameters, metadata, raw_sample, result_holder, is_train)
        if keep_sample:
            result_holder['root_hole_node_id'] = raw_sample['HoleNode']
            SeqDecoder.load_data_from_sample(hyperparameters, metadata, raw_sample, result_holder, is_train)
        return keep_sample

    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super()._init_minibatch(batch_data)
        batch_data['root_hole_node_id'] = []
        self._decoder_model.init_minibatch(batch_data)

    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        batch_finished = super()._extend_minibatch_by_sample(batch_data, sample)
        original_cg_node_offset = batch_data['cg_node_offset'] - self._get_number_of_nodes_in_graph(sample)  # The offset has already been updated in our parent...
        batch_data['root_hole_node_id'].append(sample['root_hole_node_id'] + original_cg_node_offset)
        self._decoder_model.extend_minibatch_by_sample(batch_data, sample)
        return batch_finished

    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool) -> Dict[tf.Tensor, Any]:
        minibatch = super()._finalise_minibatch(batch_data, is_train)
        write_to_minibatch(minibatch, self.placeholders['root_hole_node_id'], batch_data['root_hole_node_id'])
        self._decoder_model.finalise_minibatch(batch_data, minibatch)
        return minibatch

    # ------- These are the bits that we only need for test-time:
    def _encode_one_test_sample(self, sample_data_dict: Dict[tf.Tensor, Any]) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        return (self.sess.run(self.ops['decoder_initial_state'],
                              feed_dict=sample_data_dict),
                None)

    def _tensorise_one_test_sample(self, loaded_sample: Dict[str, Any]) -> Dict[tf.Tensor, Any]:
        test_minibatch = {}
        self._init_minibatch(test_minibatch)

        # Note that we are primarily interested in the context encoding:
        super()._extend_minibatch_by_sample(test_minibatch, loaded_sample)
        test_minibatch['root_hole_node_id'].append(loaded_sample['root_hole_node_id'])
        final_test_minibatch = super()._finalise_minibatch(test_minibatch, is_train=False)
        write_to_minibatch(final_test_minibatch, self.placeholders['root_hole_node_id'], test_minibatch['root_hole_node_id'])

        return final_test_minibatch
