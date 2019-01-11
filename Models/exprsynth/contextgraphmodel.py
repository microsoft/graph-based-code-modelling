import random
import re
from abc import abstractmethod
from collections import defaultdict, Counter
from typing import Dict, Any, Set, Optional, List

import numpy as np
import tensorflow as tf
from dpu_utils.codeutils.identifiersplitting import split_identifier_into_parts
from dpu_utils.mlutils.vocabulary import Vocabulary
from dpu_utils.codeutils.lattice.lattice import LatticeVocabulary
from dpu_utils.tfmodels.sparsegnn import SparseGGNN

from .model import Model, NO_TYPE, write_to_minibatch

BIG_NUMBER = 1e7

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 2 for (idx, char) in enumerate(ALPHABET)}  # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1
MAX_EDGE_VALUE = 50.0
USES_SUBTOKEN_EDGE_NAME = "UsesSubtoken"


def _add_per_subtoken_nodes(unsplittable_node_names: Set[str], raw_data: Dict[str, Any]) -> None:
    graph_node_labels = raw_data['ContextGraph']['NodeLabels']
    subtoken_to_using_nodes = defaultdict(set)

    max_used_node_id = 0
    for node_id, node_label in graph_node_labels.items():
        node_id = int(node_id)
        max_used_node_id = max(node_id, max_used_node_id)

        # Skip AST nodes and punctuation:
        if node_label in unsplittable_node_names:
            continue

        for subtoken in split_identifier_into_parts(node_label):
            if re.search('[a-zA-Z0-9]', subtoken):
                subtoken_to_using_nodes[subtoken].add(node_id)

    subtoken_node_id = max_used_node_id
    new_edges = []
    for subtoken, using_nodes in subtoken_to_using_nodes.items():
        subtoken_node_id += 1
        graph_node_labels[str(subtoken_node_id)] = subtoken
        new_edges.extend([(using_node_id, subtoken_node_id)
                          for using_node_id in using_nodes])

    raw_data['ContextGraph']['Edges'][USES_SUBTOKEN_EDGE_NAME] = new_edges


class ContextGraphModel(Model):
    @staticmethod
    @abstractmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = Model.get_default_hyperparameters()
        my_defaults = {
                        'max_num_cg_nodes_in_batch': 100000,

                        # Context Program Graph things:
                        'excluded_cg_edge_types': [],
                        'cg_add_subtoken_nodes': True,

                        'cg_node_label_embedding_style': 'Token',  # One of ['Token', 'CharCNN']
                        'cg_node_label_vocab_size': 10000,
                        'cg_node_label_char_length': 16,
                        "cg_node_label_embedding_size": 32,

                        'cg_node_type_vocab_size': 5000,
                        'cg_node_type_max_num': 10,
                        'cg_node_type_embedding_size': 32,

                        "cg_ggnn_layer_timesteps": [3, 1, 3, 1],
                        "cg_ggnn_residual_connections": {"1": [0], "3": [0, 1]},

                        "cg_ggnn_hidden_size": 64,
                        "cg_ggnn_use_edge_bias": False,
                        "cg_ggnn_use_edge_msg_avg_aggregation": False,
                        "cg_ggnn_use_propagation_attention": False,
                        "cg_ggnn_graph_rnn_activation": "tanh",
                        "cg_ggnn_graph_rnn_cell": "GRU",

                     }
        defaults.update(my_defaults)
        return defaults

    def __init__(self, hyperparameters, run_name: Optional[str]=None, model_save_dir: Optional[str]=None, log_save_dir: Optional[str]=None):
        super().__init__(hyperparameters, run_name, model_save_dir, log_save_dir)

    @abstractmethod
    def _make_placeholders(self, is_train: bool) -> None:
        super()._make_placeholders(is_train)

        # Placeholders for context graph:
        cg_edge_type_num = len(self.metadata['cg_edge_type_dict'])

        if self.hyperparameters['cg_node_label_embedding_style'].lower() == 'token':
            self.placeholders['cg_node_label_token_ids'] = \
                tf.placeholder(dtype=tf.int32, shape=[None], name='cg_node_label_token_ids')
        elif self.hyperparameters['cg_node_label_embedding_style'].lower() == 'charcnn':
            self.placeholders['cg_unique_label_chars'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[None, self.hyperparameters['cg_node_label_char_length']],
                               name='cg_unique_label_chars')
            self.placeholders['cg_node_label_unique_indices'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[None],
                               name='cg_node_label_unique_indices')
        else:
            raise Exception("Unknown node label embedding style '%s'!" % self.hyperparameters['cg_node_label_embedding_style'])

        self.placeholders['cg_node_type_ids'] = \
            tf.placeholder(dtype=tf.int32, shape=[None, None], name='cg_node_type_ids')
        self.placeholders['cg_node_type_ids_mask'] = \
            tf.placeholder(dtype=tf.float32, shape=[None, None], name='cg_node_type_ids_mask')

        self.placeholders['cg_adjacency_lists'] = \
            [tf.placeholder(dtype=tf.int64, shape=[None, 2], name='cg_adjacency_lists_e%s' % e)
             for e in range(cg_edge_type_num)]

        self.placeholders['cg_edge_features'] = \
            {edge_idx: tf.placeholder(dtype=tf.float32,
                                      shape=[None, features_size],
                                      name='cg_edge_%s_feature' % edge_idx)
             for edge_idx, features_size in self.metadata['cg_edge_value_sizes'].items()}

        if self.hyperparameters['cg_ggnn_use_edge_bias'] or self.hyperparameters['cg_ggnn_use_edge_msg_avg_aggregation']:
            self.placeholders['cg_num_incoming_edges_per_type'] = \
                tf.placeholder(dtype=tf.float32, shape=[None, cg_edge_type_num], name='cg_num_incoming_edges_per_type')
            self.placeholders['cg_num_outgoing_edges_per_type'] = \
                tf.placeholder(dtype=tf.float32, shape=[None, cg_edge_type_num], name='cg_num_outgoing_edges_per_type')

    def _make_parameters(self):
        super()._make_parameters()

        label_embedding_size = self.hyperparameters['cg_node_label_embedding_size']
        if self.hyperparameters['cg_node_label_embedding_style'].lower() == 'token':
            type_vocab_size = self.hyperparameters['cg_node_label_vocab_size']
            self.parameters['cg_node_label_embeddings'] = \
                tf.get_variable(name='cg_node_label_embeddings',
                                shape=[type_vocab_size, label_embedding_size],
                                initializer=tf.random_normal_initializer())
        
        type_embedding_size = self.hyperparameters['cg_node_type_embedding_size']
        if type_embedding_size > 0:
            type_vocab_size = self.hyperparameters['cg_node_type_vocab_size']
            self.parameters['cg_node_type_embeddings'] = \
                tf.get_variable(name='cg_node_type_embeddings',
                                shape=[type_vocab_size, type_embedding_size],
                                initializer=tf.random_normal_initializer())

    @abstractmethod
    def _make_model(self, is_train: bool=True) -> None:
        super()._make_model(is_train)

        # ----- Compute representation of all nodes in context graph using a GGNN:
        with tf.variable_scope("ContextGraph"):
            # (1) Compute initial embeddings for the nodes in the graph:
            initial_cg_node_embeddings = []
            if self.hyperparameters['cg_node_label_embedding_style'].lower() == 'token':
                initial_cg_node_embeddings.append(
                    self.__get_node_label_token_embeddings(
                        self.placeholders['cg_node_label_token_ids']))
            elif self.hyperparameters['cg_node_label_embedding_style'].lower() == 'charcnn':
                initial_cg_node_embeddings.append(
                    self.__get_node_label_charcnn_embeddings(
                        self.placeholders['cg_unique_label_chars'],
                        self.placeholders['cg_node_label_unique_indices']))
            else:
                raise Exception("Unknown node label embedding style '%s'!"
                                % (self.hyperparameters['cg_node_label_embedding_style'],))

            if self.hyperparameters['cg_node_type_embedding_size'] > 0:
                initial_cg_node_embeddings.append(
                    self.__get_node_type_embeddings(self.placeholders['cg_node_type_ids'],
                                                    self.placeholders['cg_node_type_ids_mask']))

            initial_cg_node_embeddings = tf.concat(initial_cg_node_embeddings, axis=-1)
            initial_node_states = tf.layers.dense(initial_cg_node_embeddings,
                                                  units=self.hyperparameters['cg_ggnn_hidden_size'],
                                                  use_bias=False,
                                                  activation=None)
            initial_node_states = tf.nn.dropout(initial_node_states,
                                                self.placeholders['dropout_keep_rate'])

            # (2) Create GGNN and pass things through it:
            ggnn_hypers = {name.replace("cg_ggnn_", "", 1): value
                           for (name, value) in self.hyperparameters.items()
                           if name.startswith("cg_ggnn_")}
            ggnn_hypers['n_edge_types'] = len(self.metadata['cg_edge_type_dict'])
            ggnn_hypers['edge_features_size'] = {edge_type_idx: value_size
                                                 for edge_type_idx, value_size in self.metadata['cg_edge_value_sizes'].items()}
            ggnn_hypers['add_backwards_edges'] = True
            ggnn = SparseGGNN(ggnn_hypers)
            self.ops['cg_node_representations'] = \
                ggnn.sparse_gnn_layer(self.placeholders['dropout_keep_rate'],
                                      initial_node_states,
                                      self.placeholders['cg_adjacency_lists'],
                                      self.placeholders.get('cg_num_incoming_edges_per_type'),
                                      self.placeholders.get('cg_num_outgoing_edges_per_type'),
                                      self.placeholders['cg_edge_features'])

    def __get_node_type_embeddings(self,
                                   node_type_ids: tf.Tensor,
                                   node_type_ids_mask: tf.Tensor) -> tf.Tensor:
        """
        :param node_type_ids: Tensor of shape [V, T] representing type of each node (allows up to T types, identified by IDs into type vocab).
        :param node_type_ids_mask: Tensor of shape [V, T] representing which type information is actually used.
        :return: Tensor of shape [V, D] representing embedded type information about each node.
        """
        node_type_embeddings = tf.nn.embedding_lookup(self.parameters['cg_node_type_embeddings'], node_type_ids)
        # Make unused type IDs /really/ small for reduce_max:
        node_type_embeddings += (1.0 - tf.expand_dims(node_type_ids_mask, axis=-1)) * -BIG_NUMBER
        node_type_embeddings = tf.reduce_max(node_type_embeddings, axis=1)  # v x D'
        return node_type_embeddings

    def __get_node_label_token_embeddings(self,
                                          node_label_token_ids: tf.Tensor) -> tf.Tensor:
        """
        :param node_label_token_ids: Tensor of shape [V] representing label of each node (identified by ID into label vocab).
        :return: Tensor of shape [V, D] representing embedded node label information about each node.
        """
        return tf.nn.embedding_lookup(self.parameters['cg_node_label_embeddings'], node_label_token_ids)

    def __get_node_label_charcnn_embeddings(self,
                                            unique_label_chars: tf.Tensor,
                                            node_label_unique_indices: tf.Tensor) -> tf.Tensor:
        """
        :param unique_label_chars: Unique labels occurring in batch
                           Shape: [num unique labels, hyperparameters['node_label_char_length']], dtype=int32
        :param node_label_unique_indices: For each node in batch, index of corresponding (unique) node label in node_label_chars_unique.
                                          Shape: [V], dtype=int32
        :return: Tensor of shape [V, D] representing embedded node label information about each node.
        """
        label_embedding_size = self.hyperparameters['cg_node_label_embedding_size']  # D
        # U ~ num unique labels
        # C ~ num characters (self.hyperparameters['node_label_char_length'])
        # A ~ num characters in alphabet
        unique_label_chars_one_hot = tf.one_hot(indices=unique_label_chars,
                                                depth=len(ALPHABET),
                                                axis=-1)  # Shape: [U, C, A]

        char_conv_l1_kernel_size = 5
        char_conv_l2_kernel_size = self.hyperparameters['cg_node_label_char_length'] - 2 * (char_conv_l1_kernel_size - 1)  # Ensures that a single value pops out

        char_conv_l1 = tf.layers.conv1d(inputs=unique_label_chars_one_hot,
                                        filters=16,
                                        kernel_size=char_conv_l1_kernel_size,
                                        activation=tf.nn.leaky_relu)     # Shape: [U, C - (char_conv_l1_kernel_size - 1), 16]
        char_pool_l1 = tf.layers.max_pooling1d(inputs=char_conv_l1,
                                               pool_size=char_conv_l1_kernel_size,
                                               strides=1)                # Shape: [U, C - 2*(char_conv_l1_kernel_size - 1), 16]

        char_conv_l2 = tf.layers.conv1d(inputs=char_pool_l1,
                                        filters=label_embedding_size,
                                        kernel_size=char_conv_l2_kernel_size,
                                        activation=tf.nn.leaky_relu)     # Shape: [U, 1, D]

        unique_label_representations = tf.squeeze(char_conv_l2, axis=1)  # Shape: [U, D]
        node_label_representations = tf.gather(params=unique_label_representations,
                                               indices=node_label_unique_indices)
        return node_label_representations

    @staticmethod
    @abstractmethod
    def _init_metadata(hyperparameters: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(ContextGraphModel, ContextGraphModel)._init_metadata(hyperparameters, raw_metadata)
        raw_metadata['cg_node_label_counter'] = Counter()
        raw_metadata['cg_node_type_counter'] = Counter()
        raw_metadata['cg_edge_types'] = set()
        raw_metadata['cg_edge_value_sizes'] = {}  # type: Dict[str, int]

    @staticmethod
    @abstractmethod
    def _load_metadata_from_sample(hyperparameters: Dict[str, Any], raw_sample: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        super(ContextGraphModel, ContextGraphModel)._load_metadata_from_sample(hyperparameters, raw_sample, raw_metadata)

        for label_token in raw_sample['ContextGraph']['NodeLabels'].values():
            raw_metadata['cg_node_label_counter'][label_token] += 1

        for type_token in raw_sample['ContextGraph']['NodeTypes'].values():
            raw_metadata['cg_node_type_counter']["type:" + type_token] += 1

        raw_metadata['cg_edge_types'].update(raw_sample['ContextGraph']['Edges'].keys())
        for edge_type, values in raw_sample['ContextGraph']['EdgeValues'].items():
            if len(values) == 0:
                continue
            existing_edge_value_size = raw_metadata['cg_edge_value_sizes'].get(edge_type)
            if existing_edge_value_size is not None:
                assert existing_edge_value_size == len(values[0])
            raw_metadata['cg_edge_value_sizes'][edge_type] = len(values[0])

    @abstractmethod
    def _finalise_metadata(self, raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super()._finalise_metadata(raw_metadata_list)

        # First, merge all needed information:
        merged_edge_types = set()
        merged_node_label_counter = Counter()
        merged_node_type_counter = Counter()
        merged_edge_value_sizes = {}
        for raw_metadata in raw_metadata_list:
            merged_edge_types.update(raw_metadata["cg_edge_types"])
            merged_node_label_counter += raw_metadata['cg_node_label_counter']
            merged_node_type_counter += raw_metadata['cg_node_type_counter']

            for edge_type, edge_value_size in raw_metadata['cg_edge_value_sizes'].items():
                existing_edge_value_size = merged_edge_value_sizes.get(edge_type)
                if existing_edge_value_size is not None:
                    assert existing_edge_value_size == edge_value_size
                merged_edge_value_sizes[edge_type] = edge_value_size

        # Store edges allowed in the context graph, and assign numerical IDs to them:
        all_used_cg_edges = list(merged_edge_types - set(self.hyperparameters['excluded_cg_edge_types']))
        if self.hyperparameters.get('cg_add_subtoken_nodes', False):
            all_used_cg_edges.append(USES_SUBTOKEN_EDGE_NAME)
        final_metadata['cg_edge_type_dict'] = {e: i for i, e in enumerate(all_used_cg_edges)}

        # Store token, type, and production vocabs:
        final_metadata['cg_node_label_vocab'] = \
            Vocabulary.create_vocabulary(
                merged_node_label_counter,
                max_size=self.hyperparameters['cg_node_label_vocab_size'])

        final_metadata['cg_node_type_vocab'] = \
            LatticeVocabulary.get_vocabulary_for(
                tokens=merged_node_type_counter,
                max_size=self.hyperparameters['cg_node_type_vocab_size'] - 1,
                lattice=final_metadata['type_lattice'])
        final_metadata['cg_node_type_vocab'].add_or_get_id(NO_TYPE)
        self.hyperparameters['cg_node_type_vocab_size'] = len(final_metadata['cg_node_type_vocab'])

        final_metadata['cg_edge_value_sizes'] = {}
        for edge_type, edge_feature_size in merged_edge_value_sizes.items():
            fwd_edge_type_idx = final_metadata['cg_edge_type_dict'][edge_type]
            final_metadata['cg_edge_value_sizes'][fwd_edge_type_idx] = edge_feature_size

        return final_metadata

    @staticmethod
    def __load_contextgraph_data_from_sample(hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                                             raw_sample: Dict[str, Any], result_holder: Dict[str, Any],
                                             is_train: bool=True) \
            -> bool:
        if hyperparameters.get('cg_add_subtoken_nodes', False):
            _add_per_subtoken_nodes(metadata['nag_reserved_names'], raw_sample)
        graph_node_labels = raw_sample['ContextGraph']['NodeLabels']
        graph_node_types = raw_sample['ContextGraph']['NodeTypes']
        num_nodes = len(graph_node_labels)
        if num_nodes >= hyperparameters['max_num_cg_nodes_in_batch']:
            print("Dropping example using %i nodes in context graph" % (num_nodes,))
            return False

        # Translate node label, either using the token vocab or into a character representation:
        if hyperparameters['cg_node_label_embedding_style'].lower() == 'token':
            # Translate node labels using the token vocabulary:
            node_labels = np.zeros((num_nodes,), dtype=np.uint16)
            for (node, label) in graph_node_labels.items():
                node_labels[int(node)] = metadata['cg_node_label_vocab'].get_id_or_unk(label)
            result_holder['cg_node_label_token_ids'] = node_labels
        elif hyperparameters['cg_node_label_embedding_style'].lower() == 'charcnn':
            # Translate node labels into character-based representation, and make unique per context graph:
            node_label_chars = np.zeros(shape=(num_nodes,
                                               hyperparameters['cg_node_label_char_length']),
                                        dtype=np.uint8)
            for (node, label) in graph_node_labels.items():
                for (char_idx, label_char) in enumerate(label[:hyperparameters['cg_node_label_char_length']].lower()):
                    node_label_chars[int(node), char_idx] = ALPHABET_DICT.get(label_char, 1)
            unique_label_chars, node_label_unique_indices = np.unique(node_label_chars,
                                                                      axis=0,
                                                                      return_inverse=True)
            result_holder['cg_unique_label_chars'] = unique_label_chars
            result_holder['cg_node_label_unique_indices'] = node_label_unique_indices
        else:
            raise Exception("Unknown node label embedding style '%s'!"
                            % hyperparameters['cg_node_label_embedding_style'])

        # Translate node types, include supertypes:
        max_num_types = hyperparameters['cg_node_type_max_num']
        node_type_labels = np.full((num_nodes, max_num_types),
                                   metadata['cg_node_type_vocab'].get_id_or_unk(NO_TYPE)[0], dtype=np.uint16)
        node_type_labels_mask = np.zeros((num_nodes, max_num_types), dtype=np.bool)
        node_type_labels_mask[:, 0] = True
        for node_id, token_type in graph_node_types.items():
            node_id = int(node_id)
            node_types = metadata['cg_node_type_vocab'].get_id_or_unk('type:' + token_type, metadata['type_lattice'])
            if is_train and len(node_types) > max_num_types:
                random.shuffle(node_types,
                               random=random.random)  # Shuffle the types so that we get a mixture of the type hierarchy and not always the same ones
            node_types = node_types[:max_num_types]
            num_types = len(node_types)
            node_type_labels[node_id, :num_types] = node_types
            node_type_labels_mask[node_id, :num_types] = True

        result_holder['cg_node_type_labels'] = node_type_labels
        result_holder['cg_node_type_labels_mask'] = node_type_labels_mask

        # Split edges according to edge_type and count their numbers:
        result_holder['cg_edges'] = [[] for _ in metadata['cg_edge_type_dict']]
        result_holder['cg_edge_values'] = {}
        num_edge_types = len(metadata['cg_edge_type_dict'])
        num_incoming_edges_per_type = np.zeros((num_nodes, num_edge_types), dtype=np.uint16)
        num_outgoing_edges_per_type = np.zeros((num_nodes, num_edge_types), dtype=np.uint16)
        for (e_type, e_type_idx) in metadata['cg_edge_type_dict'].items():
            if e_type in raw_sample['ContextGraph']['Edges']:
                edges = np.array(raw_sample['ContextGraph']['Edges'][e_type], dtype=np.int32)
                result_holder['cg_edges'][e_type_idx] = edges

                if e_type_idx in metadata['cg_edge_value_sizes']:
                    edge_values = np.array(raw_sample['ContextGraph']['EdgeValues'][e_type], dtype=np.float32)
                    edge_values = np.array(edge_values, dtype=np.float32)
                    result_holder['cg_edge_values'][e_type_idx] = \
                        np.clip(edge_values, -MAX_EDGE_VALUE, MAX_EDGE_VALUE) / MAX_EDGE_VALUE
            else:
                result_holder['cg_edges'][e_type_idx] = np.zeros((0, 2), dtype=np.int32)
                if e_type_idx in metadata['cg_edge_value_sizes']:
                    result_holder['cg_edge_values'][e_type_idx] = \
                        np.zeros((0, metadata['cg_edge_value_sizes'][e_type_idx]), dtype=np.float32)
            num_incoming_edges_per_type[:, e_type_idx] = np.bincount(result_holder['cg_edges'][e_type_idx][:, 1],
                                                                     minlength=num_nodes)
            num_outgoing_edges_per_type[:, e_type_idx] = np.bincount(result_holder['cg_edges'][e_type_idx][:, 0],
                                                                     minlength=num_nodes)
        result_holder['cg_num_incoming_edges_per_type'] = num_incoming_edges_per_type
        result_holder['cg_num_outgoing_edges_per_type'] = num_outgoing_edges_per_type

        return True

    @staticmethod
    @abstractmethod
    def _load_data_from_sample(hyperparameters: Dict[str, Any],
                               metadata: Dict[str, Any],
                               raw_sample: Dict[str, Any],
                               result_holder: Dict[str, Any],
                               is_train: bool=True) -> bool:
        keep_sample = super(ContextGraphModel, ContextGraphModel)._load_data_from_sample(hyperparameters,
                                                                                         metadata,
                                                                                         raw_sample,
                                                                                         result_holder,
                                                                                         is_train)
        return keep_sample and ContextGraphModel.__load_contextgraph_data_from_sample(hyperparameters,
                                                                                      metadata,
                                                                                      raw_sample=raw_sample,
                                                                                      result_holder=result_holder,
                                                                                      is_train=is_train)

    @abstractmethod
    def _init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super()._init_minibatch(batch_data)

        batch_data['cg_node_offset'] = 0

        if self.hyperparameters['cg_node_label_embedding_style'].lower() == 'token':
            batch_data['cg_node_label_token_ids'] = []
        elif self.hyperparameters['cg_node_label_embedding_style'].lower() == 'charcnn':
            batch_data['cg_node_label_index_offset'] = 0
            batch_data['cg_unique_label_chars'] = []
            batch_data['cg_node_label_unique_indices'] = []
        else:
            raise Exception("Unknown node label embedding style '%s'!"
                            % self.hyperparameters['cg_node_label_embedding_style'])

        if self.hyperparameters['cg_node_type_embedding_size'] > 0:
            batch_data['cg_node_type_ids'] = []
            batch_data['cg_node_type_ids_mask'] = []

        batch_data['cg_adjacency_lists'] = [[] for _ in self.metadata['cg_edge_type_dict']]
        batch_data['cg_edge_values'] = {edge_type_idx: []
                                        for edge_type_idx in self.metadata['cg_edge_value_sizes'].keys()}
        batch_data['cg_num_incoming_edges_per_type'] = []
        batch_data['cg_num_outgoing_edges_per_type'] = []

    def _get_number_of_nodes_in_graph(self, sample: Dict[str, Any]) -> int:
        if self.hyperparameters['cg_node_label_embedding_style'].lower() == 'token':
            return len(sample['cg_node_label_token_ids'])
        elif self.hyperparameters['cg_node_label_embedding_style'].lower() == 'charcnn':
            return len(sample['cg_node_label_unique_indices'])
        else:
            raise Exception("Unknown node label embedding style '%s'!"
                            % self.hyperparameters['cg_node_label_embedding_style'])

    @abstractmethod
    def _extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> bool:
        super()._extend_minibatch_by_sample(batch_data, sample)

        if self.hyperparameters['cg_node_label_embedding_style'].lower() == 'token':
            batch_data['cg_node_label_token_ids'].extend(sample['cg_node_label_token_ids'])
        elif self.hyperparameters['cg_node_label_embedding_style'].lower() == 'charcnn':
            # As we keep adding new "unique" labels, we need to shift the indices we are referring accordingly:
            batch_data['cg_unique_label_chars'].extend(sample['cg_unique_label_chars'])
            batch_data['cg_node_label_unique_indices'].extend(
                sample['cg_node_label_unique_indices'] + batch_data['cg_node_label_index_offset'])
            batch_data['cg_node_label_index_offset'] += len(sample['cg_unique_label_chars'])
        else:
            raise Exception("Unknown node label embedding style '%s'!"
                            % self.hyperparameters['cg_node_label_embedding_style'])

        if self.hyperparameters['cg_node_type_embedding_size'] > 0:
            batch_data['cg_node_type_ids'].extend(sample['cg_node_type_labels'])
            batch_data['cg_node_type_ids_mask'].extend(sample['cg_node_type_labels_mask'])

        batch_data['cg_num_incoming_edges_per_type'].extend(sample['cg_num_incoming_edges_per_type'])
        batch_data['cg_num_outgoing_edges_per_type'].extend(sample['cg_num_outgoing_edges_per_type'])
        for edge_type in self.metadata['cg_edge_type_dict'].values():
            batch_data['cg_adjacency_lists'][edge_type].extend(sample['cg_edges'][edge_type] + batch_data['cg_node_offset'])
        for edge_type, edge_values in batch_data['cg_edge_values'].items():
            edge_values.extend(sample['cg_edge_values'][edge_type])

        batch_data['cg_node_offset'] += self._get_number_of_nodes_in_graph(sample)
        return batch_data['cg_node_offset'] >= self.hyperparameters['max_num_cg_nodes_in_batch']

    @abstractmethod
    def _finalise_minibatch(self, batch_data: Dict[str, Any], is_train: bool) -> Dict[tf.Tensor, Any]:
        minibatch = super()._finalise_minibatch(batch_data, is_train)

        if self.hyperparameters['cg_node_label_embedding_style'].lower() == 'token':
            write_to_minibatch(minibatch, self.placeholders['cg_node_label_token_ids'], batch_data['cg_node_label_token_ids'])
        elif self.hyperparameters['cg_node_label_embedding_style'].lower() == 'charcnn':
            write_to_minibatch(minibatch, self.placeholders['cg_unique_label_chars'], batch_data['cg_unique_label_chars'])
            write_to_minibatch(minibatch, self.placeholders['cg_node_label_unique_indices'], batch_data['cg_node_label_unique_indices'])
        else:
            raise Exception("Unknown node label embedding style '%s'!"
                            % self.hyperparameters['cg_node_label_embedding_style'])

        if self.hyperparameters['cg_node_type_embedding_size'] > 0:
            write_to_minibatch(minibatch, self.placeholders['cg_node_type_ids'], batch_data['cg_node_type_ids'])
            write_to_minibatch(minibatch, self.placeholders['cg_node_type_ids_mask'], batch_data['cg_node_type_ids_mask'])
        if self.hyperparameters['cg_ggnn_use_edge_bias'] or self.hyperparameters['cg_ggnn_use_edge_msg_avg_aggregation']:
            write_to_minibatch(minibatch, self.placeholders['cg_num_incoming_edges_per_type'], batch_data['cg_num_incoming_edges_per_type'])
            write_to_minibatch(minibatch, self.placeholders['cg_num_outgoing_edges_per_type'], batch_data['cg_num_outgoing_edges_per_type'])

        for edge_type_idx, adjacency_list in enumerate(batch_data['cg_adjacency_lists']):
            write_to_minibatch(minibatch, self.placeholders['cg_adjacency_lists'][edge_type_idx], adjacency_list)

        for edge_type_idx, edge_values in batch_data['cg_edge_values'].items():
            write_to_minibatch(minibatch, self.placeholders['cg_edge_features'][edge_type_idx], edge_values)

        return minibatch
