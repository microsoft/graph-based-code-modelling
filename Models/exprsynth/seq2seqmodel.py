import numpy as np
from typing import Dict, Any, Tuple, Optional, List

import tensorflow as tf

from exprsynth.contexttokenmodel import ContextTokenModel
from exprsynth.seqdecoder import SeqDecoder


class Seq2SeqModel(ContextTokenModel):
    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = ContextTokenModel.get_default_hyperparameters()
        decoder_defaults = SeqDecoder.get_default_hyperparameters()
        defaults.update(decoder_defaults)
        return defaults

    def __init__(self, hyperparameters, run_name: Optional[str]=None, model_save_dir: Optional[str]=None, log_save_dir: Optional[str]=None):
        super().__init__(hyperparameters, run_name, model_save_dir, log_save_dir)
        self._decoder_model = SeqDecoder(self)

    def _make_parameters(self):
        super()._make_parameters()

        hole_cx_hidden_size = self.hyperparameters['hole_rnn_layer_size'] * 2
        self.parameters['cx_representation_to_dec_hidden'] = \
            tf.get_variable(name='cx_representation_to_dec_hidden',
                            initializer=tf.glorot_uniform_initializer(),
                            shape=[hole_cx_hidden_size, self.hyperparameters['decoder_rnn_hidden_size']],
                            )

        self._decoder_model.make_parameters()

    def _make_placeholders(self, is_train: bool) -> None:
        super()._make_placeholders(is_train)
        self._decoder_model.make_placeholders(is_train)

    def _make_model(self, is_train: bool=True):
        super()._make_model(is_train)

        # Linear layer to uncouple the latent space and dimensionality of encoder and decoder:
        num_cx_tokens_per_side = self.hyperparameters['num_cx_tokens_per_side']
        cx_hole_representations = self.ops['cx_hole_context_representations'][num_cx_tokens_per_side]
        self.ops['decoder_initial_state'] = tf.matmul(cx_hole_representations,
                                                      self.parameters['cx_representation_to_dec_hidden'])

        self._decoder_model.make_model(is_train)

    @staticmethod
    def _init_metadata(hyperparameters: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(Seq2SeqModel, Seq2SeqModel)._init_metadata(hyperparameters, raw_metadata)
        SeqDecoder.init_metadata(raw_metadata)

    @staticmethod
    def _load_metadata_from_sample(hyperparameters: Dict[str, Any], raw_sample: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(Seq2SeqModel, Seq2SeqModel)._load_metadata_from_sample(hyperparameters, raw_sample, raw_metadata)
        SeqDecoder.load_metadata_from_sample(raw_sample, raw_metadata)

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
        keep_sample = super(Seq2SeqModel, Seq2SeqModel)._load_data_from_sample(hyperparameters, metadata, raw_sample, result_holder, is_train)
        if keep_sample:
            SeqDecoder.load_data_from_sample(hyperparameters, metadata, raw_sample, result_holder, is_train)
        return keep_sample

    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super()._init_minibatch(batch_data)
        self._decoder_model.init_minibatch(batch_data)

    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        batch_finished = super()._extend_minibatch_by_sample(batch_data, sample)
        self._decoder_model.extend_minibatch_by_sample(batch_data, sample)
        return batch_finished

    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool) -> Dict[tf.Tensor, Any]:
        minibatch = super()._finalise_minibatch(batch_data, is_train)
        self._decoder_model.finalise_minibatch(batch_data, minibatch, is_train)
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
        final_test_minibatch = super()._finalise_minibatch(test_minibatch, is_train=False)

        return final_test_minibatch