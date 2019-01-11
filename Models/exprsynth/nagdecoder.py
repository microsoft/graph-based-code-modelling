from collections import defaultdict, Counter, OrderedDict, namedtuple, deque
from typing import List, Dict, Any, Tuple, Iterable, Set, Optional

import numpy as np
import tensorflow as tf
from dpu_utils.tfutils import unsorted_segment_logsumexp, pick_indices_from_probs
from dpu_utils.mlutils.vocabulary import Vocabulary
from dpu_utils.tfmodels import AsyncGGNN

from .model import Model, ModelTestResult, write_to_minibatch

BIG_NUMBER = 1e7
SMALL_NUMBER = 1e-7

# These are special non-terminal symbols that are expanded to literals, either from
# a small dict that we collect during training, or by copying from the context.
ROOT_NONTERMINAL = 'Expression'
VARIABLE_NONTERMINAL = 'Variable'
LITERAL_NONTERMINALS = ['IntLiteral', 'CharLiteral', 'StringLiteral']
LAST_USED_TOKEN_NAME = '<LAST TOK>'
EXPANSION_LABELED_EDGE_TYPE_NAMES = ["Child"]
EXPANSION_UNLABELED_EDGE_TYPE_NAMES = ["Parent", "NextUse", "NextToken", "NextSibling", "NextSubtree",
                                       "InheritedToSynthesised"]


class MissingProductionException(Exception):
    pass


ExpansionInformation = namedtuple("ExpansionInformation", ["node_to_type", "node_to_label", "node_to_prod_id", "node_to_children", "node_to_parent",
                                                           "node_to_synthesised_attr_node", "node_to_inherited_attr_node",
                                                           "variable_to_last_use_id", "node_to_representation",
                                                           "node_to_labeled_incoming_edges", "node_to_unlabeled_incoming_edges",
                                                           "context_token_representations", "context_token_mask", "context_tokens",
                                                           "literal_production_choice_normalizer",
                                                           "nodes_to_expand", "expansion_logprob", "num_expansions",
                                                           ])


def clone_list_defaultdict(dict_to_clone) -> defaultdict:
    return defaultdict(list, {key: list(value) for (key, value) in dict_to_clone.items()})


def clone_expansion_info(expansion_info: ExpansionInformation, increment_expansion_counter: bool=False) -> ExpansionInformation:
    """Create a clone of the expansion information with fresh copies of all (morally) mutable components."""
    return ExpansionInformation(node_to_type=dict(expansion_info.node_to_type),
                                node_to_label=dict(expansion_info.node_to_label),
                                node_to_prod_id=dict(expansion_info.node_to_prod_id),
                                node_to_children=clone_list_defaultdict(expansion_info.node_to_children),
                                node_to_parent=dict(expansion_info.node_to_parent),
                                node_to_synthesised_attr_node=dict(expansion_info.node_to_synthesised_attr_node),
                                node_to_inherited_attr_node=dict(expansion_info.node_to_inherited_attr_node),
                                variable_to_last_use_id=dict(expansion_info.variable_to_last_use_id),
                                node_to_representation=dict(expansion_info.node_to_representation),
                                node_to_labeled_incoming_edges=dict(expansion_info.node_to_labeled_incoming_edges),
                                node_to_unlabeled_incoming_edges=dict(expansion_info.node_to_unlabeled_incoming_edges),
                                context_token_representations=expansion_info.context_token_representations,
                                context_token_mask=expansion_info.context_token_mask,
                                context_tokens=expansion_info.context_tokens,
                                literal_production_choice_normalizer=expansion_info.literal_production_choice_normalizer,
                                nodes_to_expand=deque(expansion_info.nodes_to_expand),
                                expansion_logprob=[expansion_info.expansion_logprob[0]],
                                num_expansions=expansion_info.num_expansions + (1 if increment_expansion_counter else 0))


def get_tokens_from_expansion(expansion_info: ExpansionInformation, root_node: int) -> List[str]:
    def dfs(node: int) -> List[str]:
        children = expansion_info.node_to_children.get(node)
        if children is None:
            return [expansion_info.node_to_label[node]]
        else:
            return [tok
                    for child in children
                    for tok in dfs(child)]

    return dfs(root_node)


def raw_rhs_to_tuple(symbol_to_kind, symbol_to_label, rhs_ids: Iterable[int]) -> Tuple:
    rhs = []
    for rhs_node_id in rhs_ids:
        rhs_node_kind = symbol_to_kind[str(rhs_node_id)]
        if rhs_node_kind == "Token":
            rhs.append(symbol_to_label[str(rhs_node_id)])
        else:
            rhs.append(rhs_node_kind)
    return tuple(rhs)


def get_restricted_edge_types(hyperparameters: Dict[str, Any]):
    expansion_labeled_edge_types = OrderedDict(
        (name, edge_id) for (edge_id, name) in enumerate(n for n in EXPANSION_LABELED_EDGE_TYPE_NAMES
                                                         if n not in hyperparameters.get('exclude_edge_types', [])))
    expansion_unlabeled_edge_types = OrderedDict(
        (name, edge_id) for (edge_id, name) in enumerate(n for n in EXPANSION_UNLABELED_EDGE_TYPE_NAMES
                                                         if n not in hyperparameters.get('exclude_edge_types', [])))
    return expansion_labeled_edge_types, expansion_unlabeled_edge_types


class NAGDecoder(object):
    """
    Class implementing Neural Attribute Grammar decoding.

    It is important to note that the training code (batched) and testing code (non-batched, to
    ease beam search) are fairly independent.

    At train time, we know the target values for all nodes in the graph, and hence can
    compute representations for all nodes in one go (using a standard AsyncGNN).
    We then read out the representation for all nodes at which we make predictions,
    pass them through a layer to obtain logits for these, and then use cross-entropy
    loss to optimize them.

    At test time, we need to deal with a dynamic structure of the graph. To handle this,
    we query the NN step-wise, using three kinds of operations:
    * Context + Initialization: This produces the representation of the first few nodes
      in the expansion graph (i.e., corresponding to the last variable / token / root
      node).
    * Message passing: This does one-step propagation in the AysncGNN (exposed as
      operation 'eg_step_propagation_result'), producing the representation of one
      new node.
    * Predict expansion: Project a node representation to a decision how to expand
      it (exposed as ops 'eg_*production_choice_probs').
    All the construction of edges, sampling, beam search, etc. is handled in pure
    Python.

    In comments and documentation in this class, shape annotations are often provided, in which
    abbreviations are used:
    - B ~ "Batch size": Number of NAG graphs in the current batch (corresponds to number
      of contexts).
    - GP ~ "Grammar Productions": Number of NAG graph nodes at which we need to choose a
      production from the grammar.
    - VP ~ "Variable Productions": Number of NAG graph nodes at which we need to choose a
      variable from the context.
    - LP ~ "Literal Productions": Number of NAG graph nodes at which we need to choose a
      literal from the context.
    - D ~ "Dimension": Dimension of the node representations in the core (async) GNN.
      Hyperparameter "eg_hidden_size".
    """
    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        defaults = {
                    'eg_token_vocab_size': 100,
                    'eg_literal_vocab_size': 10,
                    'eg_max_variable_choices': 10,
                    'eg_propagation_substeps': 50,
                    'eg_hidden_size': 64,
                    'eg_edge_label_size': 16,
                    'exclude_edge_types': [],

                    'eg_graph_rnn_cell': 'GRU',  # GRU or RNN
                    'eg_graph_rnn_activation': 'tanh',  # tanh, ReLU

                    'eg_use_edge_bias': False,

                    'eg_use_vars_for_production_choice': True,  # Use mean-pooled variable representation as input for production choice
                    'eg_update_last_variable_use_representation': True,

                    'eg_use_literal_copying': True,
                    'eg_use_context_attention': True,
                    'eg_max_context_tokens': 500,
                   }
        return defaults

    def __init__(self, context_model: Model):
        # We simply share all internal data structures with the context model, so store it:
        self.__context_model = context_model
        pass

    @property
    def hyperparameters(self):
        return self.__context_model.hyperparameters

    @property
    def metadata(self):
        return self.__context_model.metadata

    @property
    def parameters(self):
        return self.__context_model.parameters

    @property
    def placeholders(self):
        return self.__context_model.placeholders

    @property
    def ops(self):
        return self.__context_model.ops

    @property
    def sess(self):
        return self.__context_model.sess

    def train_log(self, msg) -> None:
        return self.__context_model.train_log(msg)

    def test_log(self, msg) -> None:
        return self.__context_model.test_log(msg)

    # ---------- Constructing the core model ----------
    def make_parameters(self):
        # Use an OrderedDict so that we can rely on the iteration order later.
        self.__expansion_labeled_edge_types, self.__expansion_unlabeled_edge_types = \
            get_restricted_edge_types(self.hyperparameters)

        eg_token_vocab_size = len(self.metadata['eg_token_vocab'])
        eg_hidden_size = self.hyperparameters['eg_hidden_size']
        eg_edge_label_size = self.hyperparameters['eg_edge_label_size']
        self.parameters['eg_token_embeddings'] = \
            tf.get_variable(name='eg_token_embeddings',
                            shape=[eg_token_vocab_size, eg_hidden_size],
                            initializer=tf.random_normal_initializer(),
                            )

        # TODO: Should be more generic than being fixed to the number productions...
        if eg_edge_label_size > 0:
            self.parameters['eg_edge_label_embeddings'] = \
                tf.get_variable(name='eg_edge_label_embeddings',
                                shape=[len(self.metadata['eg_edge_label_vocab']), eg_edge_label_size],
                                initializer=tf.random_normal_initializer(),
                                )

    def make_placeholders(self, is_train: bool) -> None:
        self.placeholders['eg_node_token_ids'] = tf.placeholder(tf.int32,
                                                                [None],
                                                                name="eg_node_token_ids")

        # List of lists of lists of (embeddings of) labels of edges L_{r,s,e}: Labels of edges of type
        # e propagating in step s.
        # Restrictions: len(L_{s,e}) = len(S_{s,e})  [see __make_train_placeholders]
        self.placeholders['eg_edge_label_ids'] = \
            [[tf.placeholder(dtype=tf.int32,
                             shape=[None],
                             name="eg_edge_label_step%i_typ%i" % (step, edge_typ))
              for edge_typ in range(len(self.__expansion_labeled_edge_types))]
             for step in range(self.hyperparameters['eg_propagation_substeps'])]

        if is_train:
            self.__make_train_placeholders()
        else:
            self.__make_test_placeholders()

    def __make_train_placeholders(self):
        eg_edge_type_num = len(self.__expansion_labeled_edge_types) + len(self.__expansion_unlabeled_edge_types)
        # Initial nodes I: Node IDs that will have no (active) incoming edges.
        self.placeholders['eg_initial_node_ids'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name="eg_initial_node_ids")

        # Sending nodes S_{s,e}: Source node ids of edges of type e propagating in step s.
        # Restrictions: If v in S_{s,e}, then v in R_{s'} for s' < s or v in I.
        self.placeholders['eg_sending_node_ids'] = \
            [[tf.placeholder(dtype=tf.int32,
                             shape=[None],
                             name="eg_sending_node_ids_step%i_edgetyp%i" % (step, edge_typ))
              for edge_typ in range(eg_edge_type_num)]
             for step in range(self.hyperparameters['eg_propagation_substeps'])]

        # Normalised edge target nodes T_{s}: Targets of edges propagating in step s, normalised to a
        # continuous range starting from 0. This is used for aggregating messages from the sending nodes.
        self.placeholders['eg_msg_target_node_ids'] = \
            [tf.placeholder(dtype=tf.int32,
                            shape=[None],
                            name="eg_msg_targets_nodes_step%i" % (step,))
             for step in range(self.hyperparameters['eg_propagation_substeps'])]

        # Receiving nodes R_{s}: Target node ids of aggregated messages in propagation step s.
        # Restrictions: If v in R_{s}, v not in R_{s'} for all s' != s and v not in I
        self.placeholders['eg_receiving_node_ids'] = \
            [tf.placeholder(dtype=tf.int32,
                            shape=[None],
                            name="eg_receiving_nodes_step%i" % (step,))
             for step in range(self.hyperparameters['eg_propagation_substeps'])]

        # Number of receiving nodes N_{s}
        # Restrictions: N_{s} = len(R_{s})
        self.placeholders['eg_receiving_node_nums'] = \
            tf.placeholder(dtype=tf.int32,
                           shape=[self.hyperparameters['eg_propagation_substeps']],
                           name="eg_receiving_nodes_nums")

        self.placeholders['eg_production_nodes'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name="eg_production_nodes")

        if self.hyperparameters['eg_use_vars_for_production_choice']:
            self.placeholders['eg_production_var_last_use_node_ids'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[None],
                               name="eg_production_var_last_use_node_ids")
            self.placeholders['eg_production_var_last_use_node_ids_target_ids'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[None],
                               name="eg_production_var_last_use_node_ids_target_ids")

        self.placeholders['eg_production_node_choices'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name="eg_production_node_choices")

        if self.hyperparameters['eg_use_context_attention']:
            self.placeholders['eg_production_to_context_id'] = \
                tf.placeholder(dtype=tf.int32, shape=[None], name="eg_production_to_context_id")

        self.placeholders['eg_varproduction_nodes'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name='eg_varproduction_nodes')

        self.placeholders['eg_varproduction_options_nodes'] = \
            tf.placeholder(dtype=tf.int32,
                           shape=[None, self.hyperparameters['eg_max_variable_choices']],
                           name='eg_varproduction_options_nodes')
        self.placeholders['eg_varproduction_options_mask'] = \
            tf.placeholder(dtype=tf.float32,
                           shape=[None, self.hyperparameters['eg_max_variable_choices']],
                           name='eg_varproduction_options_mask')
        self.placeholders['eg_varproduction_node_choices'] = \
            tf.placeholder(dtype=tf.int32,
                           shape=[None],
                           name='eg_varproduction_node_choices')
        self.placeholders['eg_litproduction_nodes'] = {}
        self.placeholders['eg_litproduction_node_choices'] = {}
        self.placeholders['eg_litproduction_to_context_id'] = {}
        self.placeholders['eg_litproduction_choice_normalizer'] = {}
        for literal_kind in LITERAL_NONTERMINALS:
            self.placeholders['eg_litproduction_nodes'][literal_kind] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[None],
                               name="eg_litproduction_nodes_%s" % literal_kind)
            self.placeholders['eg_litproduction_node_choices'][literal_kind] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[None],
                               name="eg_litproduction_node_choices_%s" % literal_kind)
            if self.hyperparameters['eg_use_literal_copying']:
                self.placeholders['eg_litproduction_to_context_id'][literal_kind] = \
                    tf.placeholder(dtype=tf.int32,
                                   shape=[None],
                                   name="eg_litproduction_to_context_id_%s" % literal_kind)
                self.placeholders['eg_litproduction_choice_normalizer'][literal_kind] = \
                    tf.placeholder(dtype=tf.int32,
                                   shape=[None],
                                   name="eg_litproduction_choice_normalizer_%s" % literal_kind)

    def __make_test_placeholders(self):
        eg_h_dim = self.hyperparameters['eg_hidden_size']
        self.placeholders['eg_production_node_representation'] = \
            tf.placeholder(dtype=tf.float32,
                           shape=[eg_h_dim],
                           name="eg_production_node_representation")

        if self.hyperparameters['eg_use_vars_for_production_choice']:
            self.placeholders['eg_production_var_representations'] = \
                tf.placeholder(dtype=tf.float32,
                               shape=[None, eg_h_dim],
                               name="eg_production_var_representations")

        if self.hyperparameters["eg_use_literal_copying"] or self.hyperparameters['eg_use_context_attention']:
            self.placeholders['context_token_representations'] = \
                tf.placeholder(dtype=tf.float32,
                               shape=[self.hyperparameters['eg_max_context_tokens'], eg_h_dim],
                               name='context_token_representations')

        self.placeholders['eg_varproduction_node_representation'] = \
            tf.placeholder(dtype=tf.float32,
                           shape=[eg_h_dim],
                           name="eg_varproduction_node_representation")
        self.placeholders['eg_num_variable_choices'] = \
            tf.placeholder(dtype=tf.int32, shape=[], name='eg_num_variable_choices')
        self.placeholders['eg_varproduction_options_representations'] = \
            tf.placeholder(dtype=tf.float32,
                           shape=[None, eg_h_dim],
                           name="eg_varproduction_options_representations")

        self.placeholders['eg_litproduction_node_representation'] = \
            tf.placeholder(dtype=tf.float32,
                           shape=[eg_h_dim],
                           name="eg_litproduction_node_representation")
        if self.hyperparameters['eg_use_literal_copying']:
            self.placeholders['eg_litproduction_choice_normalizer'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[None],
                               name="eg_litproduction_choice_normalizer")

        # Used for one-step message propagation of expansion graph:
        eg_edge_type_num = len(self.__expansion_labeled_edge_types) + len(self.__expansion_unlabeled_edge_types)
        self.placeholders['eg_msg_source_representations'] = \
            [tf.placeholder(dtype=tf.float32,
                            shape=[None, eg_h_dim],
                            name="eg_msg_source_representations_etyp%i" % (edge_typ,))
             for edge_typ in range(eg_edge_type_num)]

        self.placeholders['eg_msg_target_label_id'] =\
            tf.placeholder(dtype=tf.int32, shape=[], name='eg_msg_target_label_id')

    def make_model(self, is_train: bool=True):
        if is_train:
            self.__make_train_model()
        else:
            self.__make_test_model()

    def __embed_edge_labels(self, num_eg_substeps: int) -> List[List[tf.Tensor]]:
        edge_labels = []
        for step in range(num_eg_substeps):
            step_edge_labels = []
            for edge_typ in range(len(self.__expansion_labeled_edge_types)):
                if self.hyperparameters['eg_edge_label_size'] > 0:
                    edge_label_embeddings = \
                        tf.nn.embedding_lookup(self.parameters['eg_edge_label_embeddings'],
                                               self.placeholders['eg_edge_label_ids'][step][edge_typ])
                else:
                    edge_label_embeddings = \
                        tf.zeros(shape=[tf.shape(self.placeholders['eg_edge_label_ids'][step][edge_typ])[0],
                                        0])
                step_edge_labels.append(edge_label_embeddings)
            edge_labels.append(step_edge_labels)
        return edge_labels

    def __make_train_model(self):
        # Pick CG representation where possible, and use embedding otherwise:
        eg_node_label_embeddings = \
            tf.nn.embedding_lookup(self.parameters['eg_token_embeddings'],
                                   self.placeholders['eg_node_token_ids'])
        eg_initial_node_representations = \
            tf.where(condition=self.ops['eg_node_representation_use_from_context'],
                     x=self.ops['eg_node_representations_from_context'],
                     y=eg_node_label_embeddings)

        # ----- (3) Compute representations of expansion graph using an async GNN submodel:
        eg_h_dim = self.hyperparameters['eg_hidden_size']
        eg_hypers = {name.replace("eg_", "", 1): value
                     for (name, value) in self.hyperparameters.items()
                     if name.startswith("eg_")}
        eg_hypers['propagation_rounds'] = 1
        eg_hypers['num_labeled_edge_types'] = len(self.__expansion_labeled_edge_types)
        eg_hypers['num_unlabeled_edge_types'] = len(self.__expansion_unlabeled_edge_types)
        with tf.variable_scope("ExpansionGraph"):
            eg_model = AsyncGGNN(eg_hypers)

            # Note that we only use a single async schedule here, so every argument is wrapped in
            # [] to use the generic code supporting many schedules:
            eg_node_representations = \
                eg_model.async_ggnn_layer(
                    eg_initial_node_representations,
                    [self.placeholders['eg_initial_node_ids']],
                    [self.placeholders['eg_sending_node_ids']],
                    [self.__embed_edge_labels(self.hyperparameters['eg_propagation_substeps'])],
                    [self.placeholders['eg_msg_target_node_ids']],
                    [self.placeholders['eg_receiving_node_ids']],
                    [self.placeholders['eg_receiving_node_nums']])

        # ----- (4) Finally, try to predict the right productions:
        # === Grammar productions:
        eg_production_node_representations = \
            tf.gather(params=eg_node_representations,
                      indices=self.placeholders['eg_production_nodes'])  # Shape [num_choice_nodes, D]

        if self.hyperparameters['eg_use_vars_for_production_choice']:
            variable_representations_at_prod_choice = \
                tf.gather(params=eg_node_representations,
                          indices=self.placeholders['eg_production_var_last_use_node_ids'])
            variable_representations_at_prod_choice = \
                tf.unsorted_segment_mean(
                    data=variable_representations_at_prod_choice,
                    segment_ids=self.placeholders['eg_production_var_last_use_node_ids_target_ids'],
                    num_segments=tf.shape(eg_production_node_representations)[0])
        else:
            variable_representations_at_prod_choice = None

        eg_production_choice_logits = \
            self.__make_production_choice_logits_model(
                eg_production_node_representations,
                variable_representations_at_prod_choice,
                self.ops.get('context_token_representations'),
                self.placeholders.get('context_token_mask'),
                self.placeholders.get('eg_production_to_context_id'))

        # === Variable productions
        eg_varproduction_node_representations = \
            tf.gather(params=eg_node_representations,
                      indices=self.placeholders['eg_varproduction_nodes'])  # Shape: [VP, D]
        eg_varproduction_options_nodes_flat = \
            tf.reshape(self.placeholders['eg_varproduction_options_nodes'],
                       shape=[-1])  # Shape [VP * eg_max_variable_choices]
        eg_varproduction_options_representations = \
            tf.reshape(tf.gather(params=eg_node_representations,
                                 indices=eg_varproduction_options_nodes_flat
                                 ),  # Shape: [VP * eg_max_variable_choices, D]
                       shape=[-1, self.hyperparameters['eg_max_variable_choices'], eg_h_dim]
                       )  # Shape: [VP, eg_max_variable_choices, D]
        eg_varproduction_choice_logits = \
            self.__make_variable_choice_logits_model(
                eg_varproduction_node_representations,
                eg_varproduction_options_representations,
                )  # Shape: [VP, eg_max_variable_choices]
        # Mask out unused choice options out:
        eg_varproduction_choice_logits += \
            (1.0 - self.placeholders['eg_varproduction_options_mask']) * -BIG_NUMBER

        # === Literal productions
        literal_logits = {}
        for literal_kind in LITERAL_NONTERMINALS:
            eg_litproduction_representation = \
                tf.gather(params=eg_node_representations,
                          indices=self.placeholders['eg_litproduction_nodes'][literal_kind]
                          )  # Shape: [LP, D]
            eg_litproduction_to_context_id, eg_litproduction_choice_normalizer = None, None
            if self.hyperparameters['eg_use_literal_copying']:
                eg_litproduction_to_context_id = \
                    self.placeholders['eg_litproduction_to_context_id'][literal_kind]
                eg_litproduction_choice_normalizer = \
                    self.placeholders['eg_litproduction_choice_normalizer'][literal_kind]

            literal_logits[literal_kind] = \
                self.__make_literal_choice_logits_model(
                    literal_kind,
                    eg_litproduction_representation,
                    self.ops.get('context_token_representations'),
                    self.placeholders.get('context_token_mask'),
                    eg_litproduction_to_context_id,
                    eg_litproduction_choice_normalizer,
                    )

        # (5) Compute loss:
        raw_prod_loss = \
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=eg_production_choice_logits,
                labels=self.placeholders['eg_production_node_choices'])
        raw_var_loss = \
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=eg_varproduction_choice_logits,
                labels=self.placeholders['eg_varproduction_node_choices'])

        # Normalize all losses by number of actual decisions made, which differ from batch to batch.
        # Can't use tf.reduce_mean because these can be empty, and reduce_mean gives NaN for those:
        prod_loss = tf.reduce_sum(raw_prod_loss) / (tf.cast(tf.size(raw_prod_loss), dtype=tf.float32) + SMALL_NUMBER)
        var_loss = tf.reduce_sum(raw_var_loss) / (tf.cast(tf.size(raw_var_loss), dtype=tf.float32) + SMALL_NUMBER)
        if len(LITERAL_NONTERMINALS) > 0:
            raw_lit_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=literal_logits[literal_kind],
                                labels=self.placeholders['eg_litproduction_node_choices'][literal_kind])
                            for literal_kind in LITERAL_NONTERMINALS]
            raw_lit_loss = tf.concat(raw_lit_loss, axis=0)
            lit_loss = tf.reduce_sum(raw_lit_loss) / (tf.cast(tf.size(raw_lit_loss), dtype=tf.float32) + SMALL_NUMBER)
        else:
            raw_lit_loss = [0.0]
            lit_loss = 0.0

        self.ops['loss'] = prod_loss + var_loss + lit_loss

        # TODO: If we want to batch this per sample, then we need an extra placeholder that maps productions/variables to
        # samples and use unsorted_segment_sum to gather together all the logprobs from all productions.
        self.ops['log_probs'] = -tf.reduce_sum(raw_prod_loss) - tf.reduce_sum(raw_var_loss) - tf.reduce_sum(raw_lit_loss)

    def __make_test_model(self):
        # Almost everywhere, we need to add extra dimensions to account for the fact that in training,
        # we need to handle several samples in one batch, whereas we don't do that during test:
        context_token_representations = self.placeholders.get('context_token_representations')
        context_token_masks = self.placeholders.get('context_token_mask')
        if context_token_representations is not None:
            context_token_representations = tf.expand_dims(context_token_representations, axis=0)

        # === Grammar productions:
        if self.hyperparameters['eg_use_vars_for_production_choice']:
            pooled_variable_representations_at_prod_choice = \
                tf.reduce_mean(self.placeholders['eg_production_var_representations'], axis=0)
            pooled_variable_representations_at_prod_choice = \
                tf.expand_dims(pooled_variable_representations_at_prod_choice, axis=0)
        else:
            pooled_variable_representations_at_prod_choice = None
        eg_production_choice_logits = \
            self.__make_production_choice_logits_model(
                tf.expand_dims(self.placeholders['eg_production_node_representation'], axis=0),
                pooled_variable_representations_at_prod_choice,
                context_token_representations,
                context_token_masks,
                production_to_context_id=tf.constant([0], dtype=tf.int32))
        self.ops['eg_production_choice_probs'] = tf.nn.softmax(eg_production_choice_logits)[0]

        # === Variable productions
        eg_varproduction_choice_logits = \
            self.__make_variable_choice_logits_model(
                tf.expand_dims(self.placeholders['eg_varproduction_node_representation'], axis=0),
                tf.expand_dims(self.placeholders['eg_varproduction_options_representations'], axis=0),
                )
        eg_varproduction_choice_logits = tf.squeeze(eg_varproduction_choice_logits, axis=0)
        self.ops['eg_varproduction_choice_probs'] = tf.nn.softmax(eg_varproduction_choice_logits, dim=-1)

        # === Literal productions
        self.ops['eg_litproduction_choice_probs'] = {}
        for literal_kind in LITERAL_NONTERMINALS:
            literal_logits = \
                self.__make_literal_choice_logits_model(
                    literal_kind,
                    tf.expand_dims(self.placeholders['eg_litproduction_node_representation'], axis=0),
                    context_token_representations,
                    context_token_masks,
                    tf.constant([0], dtype=tf.int32),
                    self.placeholders.get('eg_litproduction_choice_normalizer'),
                    )
            self.ops['eg_litproduction_choice_probs'][literal_kind] = \
                tf.nn.softmax(literal_logits, axis=-1)[0]

        # Expose one-step message propagation in expansion graph:
        eg_hypers = {name.replace("eg_", "", 1): value
                     for (name, value) in self.hyperparameters.items()
                     if name.startswith("eg_")}
        eg_hypers['propagation_rounds'] = 1
        eg_hypers['num_labeled_edge_types'] = len(self.__expansion_labeled_edge_types)
        eg_hypers['num_unlabeled_edge_types'] = len(self.__expansion_unlabeled_edge_types)
        with tf.variable_scope("ExpansionGraph"):
            eg_model = AsyncGGNN(eg_hypers)

            # First, embed edge labels. We only need the first step:
            edge_labels = self.__embed_edge_labels(1)[0]

            all_sending_representations = \
                tf.concat(self.placeholders['eg_msg_source_representations'], axis=0)
            msg_target_ids = tf.zeros([tf.shape(all_sending_representations)[0]], dtype=tf.int32)
            receiving_node_num = tf.constant(1, dtype=tf.int32)
            # Get node label embedding:
            target_node_label_embeddings =\
                tf.nn.embedding_lookup(
                    self.parameters['eg_token_embeddings'],
                    tf.expand_dims(self.placeholders['eg_msg_target_label_id'], axis=0),
                    )  # Shape [1, eg_h_dim]

            with tf.variable_scope("async_ggnn/prop_round0"):
                self.ops['eg_step_propagation_result'] = \
                    eg_model.propagate_one_step(self.placeholders['eg_msg_source_representations'],
                                                edge_labels,
                                                msg_target_ids,
                                                receiving_node_num,
                                                target_node_label_embeddings)[0]

    def __make_production_choice_logits_model(
            self,
            production_node_representations: tf.Tensor,
            variable_representations_at_prod_choice: Optional[tf.Tensor],
            context_representations: Optional[tf.Tensor],
            context_representations_mask: Optional[tf.Tensor],
            production_to_context_id: Optional[tf.Tensor],
            ) -> tf.Tensor:
        """
        Args:
            production_node_representations: Representations of nodes at which we choose grammar
                productions. Shape: [GP, D]
            variable_representations_at_prod_choice: [Optional] Representations of the current
                values of variables at nodes at which we choose grammar productions. Shape: [GP, D]
            context_representations: [Optional] Representations of nodes in context.
                Shape: [B, eg_max_context_tokens, D]
            context_representations_mask: [Optional] 0/1 mask marking which entries in
                context_representations are meaningful. Shape: [B, eg_max_context_tokens]
            production_to_context_id: [Optional] Map from entries in
                production_node_representations to the context. Shape: [GP]

        Returns:
            Logits for the choices of grammar production rules in the current batch.
            Shape: [GP, eg_production_num]
        """
        eg_production_choice_inputs = [production_node_representations]

        if self.hyperparameters['eg_use_vars_for_production_choice']:
            eg_production_choice_inputs.append(variable_representations_at_prod_choice)

        if self.hyperparameters.get('eg_use_context_attention', False):
            context_token_representations = \
                tf.gather(params=context_representations,
                          indices=production_to_context_id)                           # Shape [GP, eg_max_context_tokens, D]
            attention_scores = \
                tf.matmul(a=tf.expand_dims(production_node_representations, axis=1),  # Shape [GP, 1, D]
                          b=context_token_representations,                            # Shape [GP, eg_max_context_tokens, D]
                          transpose_b=True)                                           # Shape [GP, 1, eg_max_context_tokens]
            attention_scores = tf.squeeze(attention_scores, axis=1)                   # Shape [GP, eg_max_context_tokens]
            context_masks = tf.gather(params=context_representations_mask,
                                      indices=production_to_context_id)
            context_masks = (1 - context_masks) * -BIG_NUMBER
            attention_scores += context_masks

            attention_weight = tf.nn.softmax(attention_scores)
            weighted_context_token_representations = \
                context_token_representations * tf.expand_dims(attention_weight, axis=-1)  # Shape [num_choice_nodes, eg_max_context_tokens, D]
            context_representations = \
                tf.reduce_sum(weighted_context_token_representations, axis=1)              # Shape [num_choice_nodes, D]
            eg_production_choice_inputs.append(context_representations)

        eg_production_choice_input = tf.concat(eg_production_choice_inputs, axis=1)
        return tf.layers.dense(eg_production_choice_input,
                               units=self.metadata['eg_production_num'],
                               activation=None,
                               kernel_initializer=tf.glorot_uniform_initializer(),
                               name="grammar_choice_representation_to_logits",
                               )

    def __make_variable_choice_logits_model(
            self,
            varchoice_node_representation: tf.Tensor,
            varchoice_options_representations: tf.Tensor) -> tf.Tensor:
        """
        Args:
            varchoice_node_representation: Representations of nodes at which we choose
                variables. Shape: [VP, D]
            varchoice_options_representations: Representations of variables that we can
                choose at each choice node.
                Shape: [VP, num_variable_choices, D]
        Returns:
            Logits for the choices of variables in the current batch.
            Shape: [VP, num_variable_choices]
        """
        varchoice_options_inner_prod = \
            tf.einsum('sd,svd->sv',
                      varchoice_node_representation,
                      varchoice_options_representations)  # Shape: [VP, num_variable_choices]
        varchoice_node_representation_repeated = \
            tf.tile(tf.expand_dims(varchoice_node_representation, axis=-2),
                    multiples=[1,
                               tf.shape(varchoice_options_representations)[1],
                               1],
                    )  # Shape: [VP, num_variable_choices, D]
        varchoice_final_options_representations = \
            tf.concat([varchoice_node_representation_repeated,
                       varchoice_options_representations,
                       tf.expand_dims(varchoice_options_inner_prod, axis=-1)],
                      axis=-1)  # Shape: [VP, num_variable_choices, 2*D + 1]
        varchoice_logits = \
            tf.layers.dense(varchoice_final_options_representations,
                            units=1,
                            use_bias=False,
                            activation=None,
                            kernel_initializer=tf.glorot_uniform_initializer(),
                            name="varchoice_representation_to_logits",
                            )  # Shape: [VP, num_variable_choices, 1]
        return tf.squeeze(varchoice_logits, axis=-1)  # Shape: [VP, num_variable_choices]

    def __make_literal_choice_logits_model(
            self,
            literal_kind: str,
            litproduction_node_representations: tf.Tensor,
            context_representations: Optional[tf.Tensor],
            context_representations_mask: Optional[tf.Tensor],
            litproduction_to_context_id: Optional[tf.Tensor],
            litproduction_choice_normalizer: Optional[tf.Tensor]
            ) -> tf.Tensor:
        """
        Args:
            literal_kind: Kind of literal we are generating.
            litproduction_node_representations: Representations of nodes at which we choose
                literals. Shape: [LP, D]
            context_representations: [Optional] Representations of nodes in context.
                Shape: [B, eg_max_context_tokens, D]
            context_representations_mask: [Optional] 0/1 mask marking which entries in
                context_representations are meaningful. Shape: [B, eg_max_context_tokens]
            litproduction_to_context_id: [Optional] Map from entries in
                litproduction_node_representations to the context. Shape: [LP]
            litproduction_choice_normalizer: [Optional] If copying from the context
                tokens, some of them may refer to the same token (potentially one
                present in the vocab). This tensor allows normalisation in these cases,
                by assigning the same ID to all occurrences of the same token (usually
                the first one).
                Shape: [LP, eg_max_context_tokens + literal_vocab_size]
        Returns:
            Logits for the choices of literal production rules in the current batch.
            Shape:
                If using copying: [LP, literal_vocab_size + eg_max_context_tokens]
                Otherwise: [LP, literal_vocab_size]
        """
        literal_logits = \
            tf.layers.dense(litproduction_node_representations,
                            units=len(self.metadata["eg_literal_vocabs"][literal_kind]),
                            activation=None,
                            kernel_initializer=tf.glorot_uniform_initializer(),
                            name="%s_literal_choice_representation_to_logits" % literal_kind,
                            )  # Shape [LP, lit_vocab_size]

        if not self.hyperparameters['eg_use_literal_copying']:
            return literal_logits

        # Do dot product with the context tokens:
        context_token_representations = \
            tf.gather(params=context_representations,
                      indices=litproduction_to_context_id)  # Shape [LP, eg_max_context_tokens, D]

        copy_scores = \
            tf.matmul(a=tf.expand_dims(litproduction_node_representations, axis=1),  # Shape [LP, 1, D]
                      b=context_token_representations,  # Shape [LP, eg_max_context_tokens, D]
                      transpose_b=True)  # Shape [LP, 1, eg_max_context_tokens]
        copy_scores = tf.squeeze(copy_scores, axis=1)  # Shape [LP, eg_max_context_tokens]

        context_masks = tf.gather(params=context_representations_mask,
                                  indices=litproduction_to_context_id)
        context_masks = (1 - context_masks) * -BIG_NUMBER  # Mask out unused context tokens
        copy_scores += context_masks

        vocab_choices_and_copy_scores = \
            tf.concat([literal_logits, copy_scores],
                      axis=1)  # Shape [num_choice_nodes, literal_vocab_size + eg_max_context_tokens]

        # Now collapse logits relating to the same token (e.g., by showing up several times in context):
        normalized_vocab_choices_and_copy_logits = \
            unsorted_segment_logsumexp(scores=tf.reshape(vocab_choices_and_copy_scores, shape=[-1]),
                                       segment_ids=litproduction_choice_normalizer,
                                       num_segments=tf.shape(litproduction_choice_normalizer)[0])

        num_literal_options = len(self.metadata["eg_literal_vocabs"][literal_kind])
        num_literal_options += self.hyperparameters['eg_max_context_tokens']

        return tf.reshape(normalized_vocab_choices_and_copy_logits,
                          shape=[-1, num_literal_options])

    # ---------- Data loading (raw data to learning-ready data) ----------
    @staticmethod
    def init_metadata(raw_metadata: Dict[str, Any]) -> None:
        raw_metadata['eg_token_counter'] = Counter()
        raw_metadata['eg_literal_counters'] = defaultdict(Counter)
        raw_metadata['eg_production_vocab'] = defaultdict(set)

    @staticmethod
    def load_metadata_from_sample(raw_sample: Dict[str, Any], raw_metadata: Dict[str, Any]) -> None:
        symbol_id_to_kind = raw_sample['SymbolKinds']
        symbol_id_to_label = raw_sample['SymbolLabels']

        for symbol_label in symbol_id_to_label.values():
            raw_metadata['eg_token_counter'][symbol_label] += 1
        for (node_id, symbol_kind) in symbol_id_to_kind.items():
            raw_metadata['eg_token_counter'][symbol_kind] += 1
            if symbol_kind in LITERAL_NONTERMINALS:
                literal = symbol_id_to_label[node_id]
                raw_metadata['eg_literal_counters'][symbol_kind][literal] += 1

        last_token_before_hole = raw_sample['ContextGraph']['NodeLabels'][str(raw_sample['LastTokenBeforeHole'])]
        raw_metadata['eg_token_counter'][last_token_before_hole] += 1
        for (lhs_id, rhs_ids) in raw_sample['Productions'].items():
            lhs_kind = symbol_id_to_kind[str(lhs_id)]
            rhs = raw_rhs_to_tuple(symbol_id_to_kind, symbol_id_to_label, rhs_ids)
            raw_metadata['eg_production_vocab'][lhs_kind].add(rhs)

    def finalise_metadata(self, raw_metadata_list: List[Dict[str, Any]], final_metadata: Dict[str, Any]) -> None:
        # First, merge all needed information:
        merged_token_counter = Counter()
        merged_literal_counters = {literal_kind: Counter() for literal_kind in LITERAL_NONTERMINALS}
        merged_production_vocab = defaultdict(set)
        for raw_metadata in raw_metadata_list:
            merged_token_counter += raw_metadata['eg_token_counter']
            for literal_kind in LITERAL_NONTERMINALS:
                merged_literal_counters[literal_kind] += raw_metadata['eg_literal_counters'][literal_kind]
            for lhs, rhs_options in raw_metadata['eg_production_vocab'].items():
                merged_production_vocab[lhs].update(rhs_options)

        final_metadata['eg_token_vocab'] = \
            Vocabulary.create_vocabulary(merged_token_counter,
                                         max_size=self.hyperparameters['eg_token_vocab_size'])

        final_metadata["eg_literal_vocabs"] = {}
        for literal_kind in LITERAL_NONTERMINALS:
            final_metadata["eg_literal_vocabs"][literal_kind] = \
                Vocabulary.create_vocabulary(merged_literal_counters[literal_kind],
                                             count_threshold=0,
                                             max_size=self.hyperparameters['eg_literal_vocab_size'])

        next_production_id = 0
        eg_production_vocab = defaultdict(dict)
        next_edge_label_id = 0
        eg_edge_label_vocab = defaultdict(dict)
        for lhs, rhs_options in sorted(merged_production_vocab.items(), key=lambda t: t[0]):
            final_metadata['eg_token_vocab'].add_or_get_id(lhs)
            for rhs in sorted(rhs_options):
                production_id = eg_production_vocab[lhs].get(rhs)
                if production_id is None:
                    production_id = next_production_id
                    eg_production_vocab[lhs][rhs] = production_id
                    next_production_id += 1
                for (rhs_symbol_index, symbol) in enumerate(rhs):
                    final_metadata['eg_token_vocab'].add_or_get_id(symbol)
                    eg_edge_label_vocab[(production_id, rhs_symbol_index)] = next_edge_label_id
                    next_edge_label_id += 1

        final_metadata["eg_production_vocab"] = eg_production_vocab
        final_metadata["eg_edge_label_vocab"] = eg_edge_label_vocab
        final_metadata['eg_production_num'] = next_production_id

        self.train_log("Imputed grammar:")
        for lhs, rhs_options in eg_production_vocab.items():
            for rhs, idx in sorted(rhs_options.items(), key=lambda v: v[1]):
                self.train_log("  %s  -[%02i]->  %s" % (str(lhs), idx, " ".join(rhs)))
        self.train_log("Known literals:")
        for literal_kind in LITERAL_NONTERMINALS:
            self.train_log("  %s: %s" % (literal_kind, sorted(final_metadata['eg_literal_vocabs'][literal_kind].token_to_id.keys())))

    @staticmethod
    def __load_expansiongraph_data_from_sample(hyperparameters: Dict[str, Any],
                                               metadata: Dict[str, Any],
                                               raw_sample: Dict[str, Any],
                                               result_holder: Dict[str, Any],
                                               is_train: bool) -> None:
        result_holder['eg_node_labels'] = []

        # "Downwards" version of a node (depends on parents & pred's). Keys are IDs from symbol expansion record:
        node_to_inherited_id = {}  # type: Dict[int, int]
        # "Upwards" version of a node (depends on children). Keys are IDs from symbol expansion record:
        node_to_synthesised_id = {}  # type: Dict[int, int]
        # Maps variable name to the id of the node where it was last used. Keys are variable names, values are from fresh space next to symbol expansion record:
        last_used_node_id = {}  # type: Dict[str, int]

        # First, we create node ids for the bits of the context graph that we want to re-use
        # and populate the intermediate data structures with them:
        prod_root_node = min(int(v) for v in raw_sample['Productions'].keys())
        node_to_inherited_id[prod_root_node] = 0
        result_holder['eg_node_labels'].append(metadata['eg_token_vocab'].get_id_or_unk(ROOT_NONTERMINAL))

        last_used_node_id[LAST_USED_TOKEN_NAME] = -1
        node_to_synthesised_id[last_used_node_id[LAST_USED_TOKEN_NAME]] = 1
        result_holder['eg_node_labels'].append(metadata['eg_token_vocab'].get_id_or_unk(
            raw_sample['ContextGraph']['NodeLabels'][str(raw_sample['LastTokenBeforeHole'])]))

        if not is_train:
            result_holder['eg_variable_eg_node_ids'] = {}
            result_holder['eg_last_token_eg_node_id'] = node_to_synthesised_id[last_used_node_id[LAST_USED_TOKEN_NAME]]
        for var, cg_graph_var_node_id in raw_sample['LastUseOfVariablesInScope'].items():
            eg_var_node_id = len(result_holder['eg_node_labels'])
            result_holder['eg_node_labels'].append(metadata['eg_token_vocab'].get_id_or_unk(var))
            last_used_node_id[var] = -len(last_used_node_id) - 1
            node_to_synthesised_id[last_used_node_id[var]] = eg_var_node_id
            if not is_train:
                result_holder['eg_variable_eg_node_ids'][var] = eg_var_node_id

        if is_train:
            NAGDecoder.__load_expansiongraph_training_data_from_sample(hyperparameters, metadata, raw_sample, prod_root_node,
                                                                       node_to_inherited_id, node_to_synthesised_id, last_used_node_id,
                                                                       result_holder)
        else:
            def collect_tokens(node: int) -> List[str]:
                node_tokens = []
                children = raw_sample['Productions'].get(str(node)) or []
                for child_id in children:
                    if str(child_id) not in raw_sample['Productions']:
                        child_label = raw_sample['SymbolLabels'].get(str(child_id)) or raw_sample['SymbolKinds'][str(child_id)]
                        node_tokens.append(child_label)
                    else:
                        node_tokens.extend(collect_tokens(child_id))
                return node_tokens

            result_holder['eg_tokens'] = collect_tokens(prod_root_node)
            result_holder['eg_root_node'] = node_to_inherited_id[prod_root_node]

    def compute_incoming_edges(self, nonterminal_nodes: Set[str], expansion_info: ExpansionInformation, node_id: int) -> None:
        assert (node_id not in expansion_info.node_to_unlabeled_incoming_edges)
        incoming_labeled_edges = defaultdict(list)  # type: Dict[str, List[Tuple[int, int]]]
        incoming_unlabeled_edges = defaultdict(list)  # type: Dict[str, List[int]]
        is_inherited_attr_node = node_id in expansion_info.node_to_synthesised_attr_node
        if is_inherited_attr_node:
            node_type = expansion_info.node_to_type[node_id]
            node_label = expansion_info.node_to_label[node_id]
            node_parent = expansion_info.node_to_parent[node_id]

            prod_id = expansion_info.node_to_prod_id[node_parent]
            this_node_child_index = expansion_info.node_to_children[node_parent].index(node_id)
            child_edge_label = self.metadata['eg_edge_label_vocab'][(prod_id, this_node_child_index)]
            incoming_labeled_edges['Child'].append((node_parent, child_edge_label))
            if node_type == VARIABLE_NONTERMINAL:
                incoming_unlabeled_edges['NextUse'].append(
                    expansion_info.node_to_synthesised_attr_node[expansion_info.variable_to_last_use_id[node_label]])
            if node_type not in nonterminal_nodes:
                incoming_unlabeled_edges['NextToken'].append(expansion_info.node_to_synthesised_attr_node[
                                                                 expansion_info.variable_to_last_use_id[
                                                                     LAST_USED_TOKEN_NAME]])
            node_siblings = expansion_info.node_to_children[node_parent]
            node_child_idx = node_siblings.index(node_id)
            if node_child_idx > 0:
                incoming_unlabeled_edges['NextSibling'].append(
                    expansion_info.node_to_synthesised_attr_node[node_siblings[node_child_idx - 1]])
            incoming_unlabeled_edges['NextSubtree'].append(expansion_info.node_to_synthesised_attr_node[
                                                               expansion_info.variable_to_last_use_id[
                                                                   LAST_USED_TOKEN_NAME]])
        else:
            inherited_node = expansion_info.node_to_inherited_attr_node[node_id]
            for child_node in expansion_info.node_to_children[inherited_node]:
                incoming_unlabeled_edges['Parent'].append(expansion_info.node_to_synthesised_attr_node[child_node])
            incoming_unlabeled_edges['InheritedToSynthesised'].append(inherited_node)
        expansion_info.node_to_labeled_incoming_edges[node_id] = incoming_labeled_edges
        expansion_info.node_to_unlabeled_incoming_edges[node_id] = incoming_unlabeled_edges

    @staticmethod
    def __load_expansiongraph_training_data_from_sample(
            hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
            raw_sample: Dict[str, Any], prod_root_node: int, node_to_inherited_id: Dict[int, int],
            node_to_synthesised_id: Dict[int, int], last_used_node_id: Dict[str, int],
            result_holder: Dict[str, Any]) -> None:
        # Shortcuts and temp data we use during construction:
        symbol_to_kind = raw_sample['SymbolKinds']  # type: Dict[str, str]
        symbol_to_prod = raw_sample['Productions']  # type: Dict[str, List[int]]
        symbol_to_label = raw_sample['SymbolLabels']  # type: Dict[str, str]
        variables_in_scope = list(sorted(raw_sample['LastUseOfVariablesInScope'].keys()))  # type: List[str]

        # These are the things we'll use in the end:
        eg_node_id_to_prod_id = []  # type: List[Tuple[int, int]]  # Pairs of (node id, chosen production id)
        eg_node_id_to_varchoice = []  # type: List[Tuple[int, List[int], int]]  # Triples of (node id, [id of last var use], index of correct var)
        eg_node_id_to_literal_choice = defaultdict(list)  # type: Dict[str, List[Tuple[int, int]]]  # Maps literal kind to pairs of (node id, chosen literal id)
        eg_prod_node_to_var_last_uses = {}  # type: Dict[int, np.ndarray]  # Dict from production node id to [id of last var use]
        eg_schedule = []  # type: List[Dict[str, List[Tuple[int, int, Optional[int]]]]]  # edge type name to edges of that type in each expansion step.

        # We will use these to pick literals:
        eg_literal_tok_to_idx = {}
        if hyperparameters['eg_use_literal_copying']:
            eg_literal_choice_normalizer_maps = {}
            # For each literal kind, we compute a "normalizer map", which we use to identify
            # choices that correspond to the same literal (e.g., when using a literal several
            # times in context)
            for literal_kind in LITERAL_NONTERMINALS:
                # Collect all choices (vocab + things we can copy from):
                literal_vocab = metadata['eg_literal_vocabs'][literal_kind]
                literal_choices = \
                    literal_vocab.id_to_token \
                    + result_holder['context_nonkeyword_tokens'][:hyperparameters['eg_max_context_tokens']]
                first_tok_occurrences = {}
                num_choices = hyperparameters['eg_max_context_tokens'] + len(literal_vocab)
                normalizer_map = np.arange(num_choices, dtype=np.int16)
                for (token_idx, token) in enumerate(literal_choices):
                    first_occ = first_tok_occurrences.get(token)
                    if first_occ is not None:
                        normalizer_map[token_idx] = first_occ
                    else:
                        first_tok_occurrences[token] = token_idx
                eg_literal_tok_to_idx[literal_kind] = first_tok_occurrences
                eg_literal_choice_normalizer_maps[literal_kind] = normalizer_map
            result_holder['eg_literal_choice_normalizer_maps'] = eg_literal_choice_normalizer_maps
        else:
            for literal_kind in LITERAL_NONTERMINALS:
                eg_literal_tok_to_idx[literal_kind] = metadata['eg_literal_vocabs'][literal_kind].token_to_id

        # Map prods onto internal numbering and compute propagation schedule:
        def declare_new_node(sym_exp_node_id: int, is_synthesised: bool) -> int:
            new_node_id = len(result_holder['eg_node_labels'])
            node_label = symbol_to_label.get(str(sym_exp_node_id)) or symbol_to_kind[str(sym_exp_node_id)]
            result_holder['eg_node_labels'].append(metadata['eg_token_vocab'].get_id_or_unk(node_label))
            if is_synthesised:
                node_to_synthesised_id[sym_exp_node_id] = new_node_id
            else:
                node_to_inherited_id[sym_exp_node_id] = new_node_id

            return new_node_id

        def expand_node(node_id: int) -> None:
            rhs_node_ids = symbol_to_prod.get(str(node_id))
            if rhs_node_ids is None:
                # In the case that we have no children, the downwards and upwards version of our node are the same:
                node_to_synthesised_id[node_id] = node_to_inherited_id[node_id]
                return

            declare_new_node(node_id, is_synthesised=True)

            # Figure out which production id this one is is and store it away:
            rhs = raw_rhs_to_tuple(symbol_to_kind, symbol_to_label, rhs_node_ids)
            known_lhs_productions = metadata['eg_production_vocab'][symbol_to_kind[str(node_id)]]
            if rhs not in known_lhs_productions:
                raise MissingProductionException("%s -> %s" % (symbol_to_kind[str(node_id)], rhs))
            prod_id = known_lhs_productions[rhs]
            eg_node_id_to_prod_id.append((node_to_inherited_id[node_id], prod_id))
            if hyperparameters['eg_use_vars_for_production_choice']:
                eg_prod_node_to_var_last_uses[node_to_inherited_id[node_id]] = \
                    np.array([node_to_synthesised_id[last_used_node_id[varchoice]] for varchoice in variables_in_scope], dtype=np.int16)
            # print("Expanding %i using rule %i %s -> %s" % (node_to_inherited_id[node_id], prod_id,
            #                                                symbol_to_label.get(str(node_id)) or symbol_to_kind[str(node_id)],
            #                                                tuple([symbol_to_kind[str(rhs_node_id)] for rhs_node_id in rhs_node_ids])))

            # Visit all children, in left-to-right order, descending into them if needed
            last_sibling = None
            parent_inwards_edges = defaultdict(list)  # type: Dict[str, List[Tuple[int, int, Optional[int]]]]
            parent_inwards_edges['InheritedToSynthesised'].append((node_to_inherited_id[node_id],
                                                                   node_to_synthesised_id[node_id],
                                                                   None))
            for (rhs_symbol_idx, child_id) in enumerate(rhs_node_ids):
                child_inherited_id = declare_new_node(child_id, is_synthesised=False)
                child_inwards_edges = defaultdict(list)  # type: Dict[str, List[Tuple[int, int, Optional[int]]]]
                child_edge_label_id = metadata['eg_edge_label_vocab'][(prod_id, rhs_symbol_idx)]
                child_inwards_edges['Child'].append((node_to_inherited_id[node_id], child_inherited_id, child_edge_label_id))

                # Connection from left sibling, and prepare to be connected to the right sibling:
                if last_sibling is not None:
                    child_inwards_edges['NextSibling'].append((node_to_synthesised_id[last_sibling], child_inherited_id, None))
                last_sibling = child_id

                # Connection from the last generated leaf ("next action" in "A syntactic neural model for general-purpose code generation", Yin & Neubig '17):
                child_inwards_edges['NextSubtree'].append((node_to_synthesised_id[last_used_node_id[LAST_USED_TOKEN_NAME]],
                                                           child_inherited_id,
                                                           None))

                # Check if we are terminal (token) node and add appropriate edges if that's the case:
                if str(child_id) not in symbol_to_prod:
                    child_inwards_edges['NextToken'].append((node_to_synthesised_id[last_used_node_id[LAST_USED_TOKEN_NAME]],
                                                             child_inherited_id,
                                                             None))
                    last_used_node_id[LAST_USED_TOKEN_NAME] = child_id

                # If we are a variable or literal, we also need to store information to train to make the right choice:
                child_kind = symbol_to_kind[str(child_id)]
                if child_kind == VARIABLE_NONTERMINAL:
                    var_name = symbol_to_label[str(child_id)]
                    # print("  Chose variable %s" % var_name)
                    last_var_use_id = last_used_node_id[var_name]
                    cur_variable_to_last_use_ids = [node_to_synthesised_id[last_used_node_id[varchoice]] for varchoice in variables_in_scope]
                    varchoice_id = variables_in_scope.index(var_name)
                    eg_node_id_to_varchoice.append((node_to_inherited_id[node_id], cur_variable_to_last_use_ids, varchoice_id))
                    child_inwards_edges['NextUse'].append((node_to_synthesised_id[last_var_use_id], child_inherited_id, None))
                    if hyperparameters['eg_update_last_variable_use_representation']:
                        last_used_node_id[var_name] = child_id
                elif child_kind in LITERAL_NONTERMINALS:
                    literal = symbol_to_label[str(child_id)]
                    # print("  Chose literal %s" % literal)
                    literal_id = eg_literal_tok_to_idx[child_kind].get(literal)
                    # In the case that a literal is not in the vocab and not in the context, the above will return None,
                    # so map that explicitly to the id for UNK:
                    if literal_id is None:
                        literal_id = metadata['eg_literal_vocabs'][child_kind].get_id_or_unk(literal)
                    eg_node_id_to_literal_choice[child_kind].append((node_to_inherited_id[node_id], literal_id))

                # Store the edges leading to new node, recurse into it, and mark its upwards connection for later:
                eg_schedule.append(child_inwards_edges)
                expand_node(child_id)
                parent_inwards_edges['Parent'].append((node_to_synthesised_id[child_id], node_to_synthesised_id[node_id], None))
            eg_schedule.append(parent_inwards_edges)

        expand_node(prod_root_node)

        expansion_labeled_edge_types, expansion_unlabeled_edge_types = get_restricted_edge_types(hyperparameters)

        def split_schedule_step(step: Dict[str, List[Tuple[int, int, Optional[int]]]]) -> List[List[Tuple[int, int]]]:
            total_edge_types = len(expansion_labeled_edge_types) + len(expansion_unlabeled_edge_types)
            step_by_edge = [[] for _ in range(total_edge_types)]  # type: List[List[Tuple[int, int]]]
            for (label, edges) in step.items():
                edges = [(v, w) for (v, w, _) in edges]  # Strip off (optional) label:
                if label in expansion_labeled_edge_types:
                    step_by_edge[expansion_labeled_edge_types[label]] = edges
                elif label in expansion_unlabeled_edge_types:
                    step_by_edge[len(expansion_labeled_edge_types) + expansion_unlabeled_edge_types[label]] = edges
            return step_by_edge

        def edge_labels_from_schedule_step(step: Dict[str, List[Tuple[int, int, Optional[int]]]]) -> List[List[int]]:
            labels_by_edge = [[] for _ in range(len(expansion_labeled_edge_types))]  # type: List[List[int]]
            for (label, edges) in step.items():
                if label in expansion_labeled_edge_types:
                    label_ids = [l for (_, _, l) in edges]  # Keep only edge label
                    labels_by_edge[expansion_labeled_edge_types[label]] = label_ids
            return labels_by_edge

        # print("Schedule:")
        # initialised_nodes = set()
        # initialised_nodes = initialised_nodes | result_holder['eg_node_id_to_cg_node_id'].keys()
        # for step_id, expansion_step in enumerate(eg_schedule):
        #     print(" Step %i" % step_id)
        #     initialised_this_step = set()
        #     for edge_type in EXPANSION_UNLABELED_EDGE_TYPE_NAMES + EXPANSION_LABELED_EDGE_TYPE_NAMES:
        #         for (v, w, _) in expansion_step[edge_type]:
        #             assert v in initialised_nodes
        #             assert w not in initialised_nodes
        #             initialised_this_step.add(w)
        #     for newly_computed_node in initialised_this_step:
        #         node_label_id = result_holder['eg_node_labels'][newly_computed_node]
        #         print("   Node Label for %i: %i (reversed %s)"
        #               % (newly_computed_node, node_label_id, metadata['eg_token_vocab'].id_to_token[node_label_id]))
        #     for edge_type in EXPANSION_UNLABELED_EDGE_TYPE_NAMES + EXPANSION_LABELED_EDGE_TYPE_NAMES:
        #         edges = expansion_step[edge_type]
        #         if len(edges) > 0:
        #             initialised_nodes = initialised_nodes | initialised_this_step
        #             print("   %s edges: [%s]" % (edge_type,
        #                                          ", ".join("(%s -[%s]> %s)" % (v, l, w) for (v, w, l) in edges)))
        # print("Variable choices:\n %s" % (str(eg_node_id_to_varchoice)))
        # print("Literal choices: \n %s" % (str(eg_node_id_to_literal_choice)))
        if hyperparameters['eg_use_vars_for_production_choice']:
            result_holder['eg_production_node_id_to_var_last_use_node_ids'] = eg_prod_node_to_var_last_uses
        result_holder['eg_node_id_to_prod_id'] = np.array(eg_node_id_to_prod_id, dtype=np.int16)
        result_holder['eg_node_id_to_varchoice'] = eg_node_id_to_varchoice
        result_holder['eg_node_id_to_literal_choice'] = {}
        for literal_kind in LITERAL_NONTERMINALS:
            literal_choice_data = eg_node_id_to_literal_choice.get(literal_kind)
            if literal_choice_data is None:
                literal_choice_data = np.empty(shape=[0, 2], dtype=np.uint16)
            else:
                literal_choice_data = np.array(literal_choice_data, dtype=np.uint16)
            result_holder['eg_node_id_to_literal_choice'][literal_kind] = literal_choice_data
        result_holder['eg_schedule'] = [split_schedule_step(step) for step in eg_schedule]
        result_holder['eg_edge_label_ids'] = [edge_labels_from_schedule_step(step) for step in eg_schedule]

    @staticmethod
    def load_data_from_sample(hyperparameters: Dict[str, Any], metadata: Dict[str, Any], raw_sample: Dict[str, Any],
                              result_holder: Dict[str, Any], is_train: bool=True) -> bool:
        try:
            NAGDecoder.__load_expansiongraph_data_from_sample(hyperparameters, metadata,
                                                              raw_sample=raw_sample, result_holder=result_holder, is_train=is_train)
            if "eg_schedule" in result_holder and len(result_holder['eg_schedule']) >= hyperparameters['eg_propagation_substeps']:
                print("Dropping example using %i propagation steps in schedule" % (len(result_holder['eg_schedule']),))
                return False
        except MissingProductionException as e:
            print("Dropping example using unknown production rule %s" % (str(e),))
            return False
        except Exception as e:
            print("Dropping example because an error happened %s" % (str(e),))
            return False

        return True

    # ---------- Minibatch construction ----------
    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        total_edge_types = len(self.__expansion_labeled_edge_types) + len(self.__expansion_unlabeled_edge_types)

        batch_data['eg_node_offset'] = 0
        batch_data['next_step_target_node_id'] = [0 for _ in range(self.hyperparameters['eg_propagation_substeps'])]

        batch_data['eg_node_token_ids'] = []
        batch_data['eg_initial_node_ids'] = []
        batch_data['eg_sending_node_ids'] = [[[] for _ in range(total_edge_types)]
                                             for _ in range(self.hyperparameters['eg_propagation_substeps'])]
        batch_data['eg_edge_label_ids'] = [[[] for _ in range(len(self.__expansion_labeled_edge_types))]
                                           for _ in range(self.hyperparameters['eg_propagation_substeps'])]
        batch_data['eg_msg_target_node_ids'] = [[[] for _ in range(total_edge_types)]
                                                for _ in range(self.hyperparameters['eg_propagation_substeps'])]
        batch_data['eg_receiving_node_ids'] = [[] for _ in range(self.hyperparameters['eg_propagation_substeps'])]
        batch_data['eg_receiving_node_nums'] = [0 for _ in range(self.hyperparameters['eg_propagation_substeps'])]

        batch_data['eg_production_nodes'] = []
        if self.hyperparameters['eg_use_vars_for_production_choice']:
            batch_data['eg_prod_idx_offset'] = 0
            batch_data['eg_production_var_last_use_node_ids'] = []
            batch_data['eg_production_var_last_use_node_ids_target_ids'] = []
        batch_data['eg_production_node_choices'] = []
        if self.hyperparameters.get('eg_use_context_attention', False):
            batch_data['eg_production_to_context_id'] = []
        batch_data['eg_varproduction_nodes'] = []
        batch_data['eg_varproduction_options_nodes'] = []
        batch_data['eg_varproduction_options_mask'] = []
        batch_data['eg_varproduction_node_choices'] = []
        batch_data['eg_litproduction_nodes'] = {literal_kind: [] for literal_kind in LITERAL_NONTERMINALS}
        batch_data['eg_litproduction_node_choices'] = {literal_kind: [] for literal_kind in LITERAL_NONTERMINALS}
        batch_data['eg_litproduction_to_context_id'] = {literal_kind: [] for literal_kind in LITERAL_NONTERMINALS}
        batch_data['eg_litproduction_choice_normalizer'] = {literal_kind: [] for literal_kind in LITERAL_NONTERMINALS}

    def __extend_minibatch_by_expansion_graph_train_from_sample(self, batch_data: Dict[str, Any],
                                                                sample: Dict[str, Any]) -> None:
        this_sample_id = batch_data['samples_in_batch'] - 1  # Counter already incremented when we get called

        total_edge_types = len(self.__expansion_labeled_edge_types) + len(self.__expansion_unlabeled_edge_types)
        for (step_num, schedule_step) in enumerate(sample['eg_schedule']):
            eg_node_id_to_step_target_id = OrderedDict()
            for edge_type in range(total_edge_types):
                for (source, target) in schedule_step[edge_type]:
                    batch_data['eg_sending_node_ids'][step_num][edge_type].append(source + batch_data['eg_node_offset'])
                    step_target_id = eg_node_id_to_step_target_id.get(target)
                    if step_target_id is None:
                        step_target_id = batch_data['next_step_target_node_id'][step_num]
                        batch_data['next_step_target_node_id'][step_num] += 1
                        eg_node_id_to_step_target_id[target] = step_target_id
                    batch_data['eg_msg_target_node_ids'][step_num][edge_type].append(step_target_id)
            for edge_type in range(len(self.__expansion_labeled_edge_types)):
                batch_data['eg_edge_label_ids'][step_num][edge_type].extend(sample['eg_edge_label_ids'][step_num][edge_type])
            for eg_target_node_id in eg_node_id_to_step_target_id.keys():
                batch_data['eg_receiving_node_ids'][step_num].append(eg_target_node_id + batch_data['eg_node_offset'])
            batch_data['eg_receiving_node_nums'][step_num] += len(eg_node_id_to_step_target_id)

        # ----- Data related to the production choices:
        batch_data['eg_production_nodes'].extend(sample['eg_node_id_to_prod_id'][:, 0] + batch_data['eg_node_offset'])
        batch_data['eg_production_node_choices'].extend(sample['eg_node_id_to_prod_id'][:, 1])
        if self.hyperparameters['eg_use_context_attention']:
            batch_data['eg_production_to_context_id'].extend([this_sample_id] * sample['eg_node_id_to_prod_id'].shape[0])

        if self.hyperparameters['eg_use_vars_for_production_choice']:
            for (prod_index, prod_node_id) in enumerate(sample['eg_node_id_to_prod_id'][:, 0]):
                var_last_uses_at_prod_node_id = sample['eg_production_node_id_to_var_last_use_node_ids'][prod_node_id]
                batch_data['eg_production_var_last_use_node_ids'].extend(var_last_uses_at_prod_node_id + batch_data['eg_node_offset'])
                overall_prod_index = prod_index + batch_data['eg_prod_idx_offset']
                batch_data['eg_production_var_last_use_node_ids_target_ids'].extend([overall_prod_index] * len(var_last_uses_at_prod_node_id))

        for (eg_varproduction_node_id, eg_varproduction_options_node_ids, chosen_id) in sample['eg_node_id_to_varchoice']:
            batch_data['eg_varproduction_nodes'].append(eg_varproduction_node_id + batch_data['eg_node_offset'])

            # Restrict to number of choices that we want to allow, make sure we keep the correct one:
            eg_varproduction_correct_node_id = eg_varproduction_options_node_ids[chosen_id]
            eg_varproduction_distractor_node_ids = eg_varproduction_options_node_ids[:chosen_id] + eg_varproduction_options_node_ids[chosen_id + 1:]
            np.random.shuffle(eg_varproduction_distractor_node_ids)
            eg_varproduction_options_node_ids = [eg_varproduction_correct_node_id]
            eg_varproduction_options_node_ids.extend(eg_varproduction_distractor_node_ids[:self.hyperparameters['eg_max_variable_choices'] - 1])
            num_of_options = len(eg_varproduction_options_node_ids)
            if num_of_options == 0:
                raise Exception("Sample is choosing a variable from an empty set.")
            num_padding = self.hyperparameters['eg_max_variable_choices'] - num_of_options
            eg_varproduction_options_mask = [1.] * num_of_options + [0.] * num_padding
            eg_varproduction_options_node_ids = np.array(eg_varproduction_options_node_ids + [0] * num_padding)
            batch_data['eg_varproduction_options_nodes'].append(eg_varproduction_options_node_ids + batch_data['eg_node_offset'])
            batch_data['eg_varproduction_options_mask'].append(eg_varproduction_options_mask)
            batch_data['eg_varproduction_node_choices'].append(0)  # We've reordered so that the correct choice is always first

        for literal_kind in LITERAL_NONTERMINALS:
            # Shape [num_choice_nodes, 2], with (v, c) meaning that at eg node v, we want to choose literal c:
            literal_choices = sample['eg_node_id_to_literal_choice'][literal_kind]

            if self.hyperparameters['eg_use_literal_copying']:
                # Prepare normalizer. We'll use an unsorted_segment_sum on the model side, and that only operates on a flattened shape
                # So here, we repeat the normalizer an appropriate number of times, but shifting by the number of choices
                normalizer_map = sample['eg_literal_choice_normalizer_maps'][literal_kind]
                num_choices_so_far = sum(choice_nodes.shape[0] for choice_nodes in batch_data['eg_litproduction_nodes'][literal_kind])
                num_choices_this_sample = literal_choices.shape[0]
                repeated_normalizer_map = np.tile(np.expand_dims(normalizer_map, axis=0),
                                                  reps=[num_choices_this_sample, 1])
                flattened_normalizer_offsets = np.repeat((np.arange(num_choices_this_sample) + num_choices_so_far) * len(normalizer_map),
                                                         repeats=len(normalizer_map))
                normalizer_offsets = np.reshape(flattened_normalizer_offsets, [-1, len(normalizer_map)])
                batch_data['eg_litproduction_choice_normalizer'][literal_kind].append(
                    np.reshape(repeated_normalizer_map + normalizer_offsets, -1))

                batch_data['eg_litproduction_to_context_id'][literal_kind].append([this_sample_id] * literal_choices.shape[0])

            batch_data['eg_litproduction_nodes'][literal_kind].append(literal_choices[:, 0] + batch_data['eg_node_offset'])
            batch_data['eg_litproduction_node_choices'][literal_kind].append(literal_choices[:, 1])

    def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any]) -> None:
        batch_data['eg_node_token_ids'].extend(sample['eg_node_labels'])
        self.__extend_minibatch_by_expansion_graph_train_from_sample(batch_data, sample)
        batch_data['eg_node_offset'] += len(sample['eg_node_labels'])
        if self.hyperparameters['eg_use_vars_for_production_choice']:
            batch_data['eg_prod_idx_offset'] += len(sample['eg_node_id_to_prod_id'][:, 0])

    def finalise_minibatch(self, batch_data: Dict[str, Any], minibatch: Dict[tf.Tensor, Any]) -> None:
        total_edge_types = len(self.__expansion_labeled_edge_types) + len(self.__expansion_unlabeled_edge_types)

        flat_batch_keys = ['eg_node_token_ids', 'eg_initial_node_ids', 'eg_receiving_node_nums',
                           'eg_production_nodes', 'eg_production_node_choices',
                           'eg_varproduction_nodes', 'eg_varproduction_options_nodes',
                           'eg_varproduction_options_mask',
                           'eg_varproduction_node_choices']
        if self.hyperparameters['eg_use_vars_for_production_choice']:
            flat_batch_keys.extend(['eg_production_var_last_use_node_ids',
                                    'eg_production_var_last_use_node_ids_target_ids',
                                    ])
        if self.hyperparameters['eg_use_context_attention']:
            flat_batch_keys.append('eg_production_to_context_id')

        for key in flat_batch_keys:
            write_to_minibatch(minibatch, self.placeholders[key], batch_data[key])

        for step in range(self.hyperparameters['eg_propagation_substeps']):
            write_to_minibatch(minibatch,
                               self.placeholders['eg_msg_target_node_ids'][step],
                               np.concatenate(batch_data['eg_msg_target_node_ids'][step]))
            write_to_minibatch(minibatch,
                               self.placeholders['eg_receiving_node_ids'][step],
                               batch_data['eg_receiving_node_ids'][step])

            for edge_type_idx in range(total_edge_types):
                write_to_minibatch(minibatch,
                                   self.placeholders['eg_sending_node_ids'][step][edge_type_idx],
                                   batch_data['eg_sending_node_ids'][step][edge_type_idx])
            for edge_type_idx in range(len(self.__expansion_labeled_edge_types)):
                write_to_minibatch(minibatch,
                                   self.placeholders['eg_edge_label_ids'][step][edge_type_idx],
                                   batch_data['eg_edge_label_ids'][step][edge_type_idx])

        for literal_kind in LITERAL_NONTERMINALS:
            write_to_minibatch(minibatch,
                               self.placeholders['eg_litproduction_nodes'][literal_kind],
                               np.concatenate(batch_data['eg_litproduction_nodes'][literal_kind], axis=0))
            write_to_minibatch(minibatch,
                               self.placeholders['eg_litproduction_node_choices'][literal_kind],
                               np.concatenate(batch_data['eg_litproduction_node_choices'][literal_kind], axis=0))
            if self.hyperparameters['eg_use_literal_copying']:
                write_to_minibatch(minibatch,
                                   self.placeholders['eg_litproduction_to_context_id'][literal_kind],
                                   np.concatenate(batch_data['eg_litproduction_to_context_id'][literal_kind], axis=0))
                write_to_minibatch(minibatch,
                                   self.placeholders['eg_litproduction_choice_normalizer'][literal_kind],
                                   np.concatenate(batch_data['eg_litproduction_choice_normalizer'][literal_kind], axis=0))

    # ---------- Test-time code ----------
    def generate_suggestions_for_one_sample(self,
                                            test_sample: Dict[str, Any],
                                            initial_eg_node_representations: tf.Tensor,
                                            beam_size: int=3,
                                            max_decoding_steps: int=100,
                                            context_tokens: Optional[List[str]]=None,
                                            context_token_representations: Optional[tf.Tensor]=None,
                                            context_token_mask: Optional[np.ndarray]=None,
                                            ) -> ModelTestResult:
        production_id_to_production = {}  # type: Dict[int, Tuple[str, Iterable[str]]]
        for (nonterminal, nonterminal_rules) in self.metadata['eg_production_vocab'].items():
            for (expansion, prod_id) in nonterminal_rules.items():
                production_id_to_production[prod_id] = (nonterminal, expansion)

        max_used_eg_node_id = initial_eg_node_representations.shape[0]

        def declare_new_node(expansion_info: ExpansionInformation, parent_node: int, node_type: str) -> int:
            nonlocal max_used_eg_node_id
            new_node_id = max_used_eg_node_id
            new_synthesised_node_id = max_used_eg_node_id + 1
            max_used_eg_node_id += 2
            expansion_info.node_to_parent[new_node_id] = parent_node
            expansion_info.node_to_children[parent_node].append(new_node_id)
            expansion_info.node_to_type[new_node_id] = node_type
            expansion_info.node_to_label[new_node_id] = node_type
            expansion_info.node_to_label[new_synthesised_node_id] = node_type
            expansion_info.node_to_synthesised_attr_node[new_node_id] = new_synthesised_node_id
            expansion_info.node_to_inherited_attr_node[new_synthesised_node_id] = new_node_id
            return new_node_id

        def get_node_attributes(expansion_info: ExpansionInformation, node_id: int) -> tf.Tensor:
            """
            Return attributes associated with node, either from cache in expansion information or by
            calling the model to compute a representation according to the edge information in
            the expansion_info.
            """
            node_attributes = expansion_info.node_to_representation.get(node_id)
            if node_attributes is None:
                node_label = expansion_info.node_to_label[node_id]
                if node_label == VARIABLE_NONTERMINAL:
                    node_label = ROOT_NONTERMINAL
                if node_id not in expansion_info.node_to_unlabeled_incoming_edges:
                    self.compute_incoming_edges(self.metadata['eg_production_vocab'].keys(), expansion_info, node_id)
                msg_prop_data = {self.placeholders['eg_msg_target_label_id']: self.metadata['eg_token_vocab'].get_id_or_unk(node_label)}
                for labeled_edge_typ in self.__expansion_labeled_edge_types.keys():
                    source_node_ids = [v for (v, _) in expansion_info.node_to_labeled_incoming_edges[node_id][labeled_edge_typ]]
                    edge_labels = [l for (_, l) in expansion_info.node_to_labeled_incoming_edges[node_id][labeled_edge_typ]]
                    if len(source_node_ids) == 0:
                        sender_repr = np.empty(shape=[0, self.hyperparameters['eg_hidden_size']])
                    else:
                        sender_repr = [get_node_attributes(expansion_info, source_node_id) for source_node_id in source_node_ids]
                    labeled_edge_typ_idx = self.__expansion_labeled_edge_types[labeled_edge_typ]
                    msg_prop_data[self.placeholders['eg_msg_source_representations'][labeled_edge_typ_idx]] = sender_repr
                    msg_prop_data[self.placeholders['eg_edge_label_ids'][0][labeled_edge_typ_idx]] = edge_labels
                for unlabeled_edge_typ in self.__expansion_unlabeled_edge_types.keys():
                    source_node_ids = expansion_info.node_to_unlabeled_incoming_edges[node_id][unlabeled_edge_typ]
                    if len(source_node_ids) == 0:
                        sender_repr = np.empty(shape=[0, self.hyperparameters['eg_hidden_size']])
                    else:
                        sender_repr = [get_node_attributes(expansion_info, source_node_id) for source_node_id in source_node_ids]
                    shifted_edge_type_id = len(self.__expansion_labeled_edge_types) + self.__expansion_unlabeled_edge_types[unlabeled_edge_typ]
                    msg_prop_data[self.placeholders['eg_msg_source_representations'][shifted_edge_type_id]] = sender_repr
                # print("Computing attributes for %i (label %s) with following edges:" % (node_id, node_label))
                # for labeled_edge_type in EXPANSION_LABELED_EDGE_TYPE_NAMES + EXPANSION_UNLABELED_EDGE_TYPE_NAMES:
                #     edges = expansion_info.node_to_labeled_incoming_edges[node_id][labeled_edge_type]
                #     if len(edges) > 0:
                #         print(" %s edges: [%s]" % (labeled_edge_type,
                #                                    ", ".join("(%s, %s)" % (w, node_id) for w in edges)))

                node_attributes = self.sess.run(self.ops['eg_step_propagation_result'], feed_dict=msg_prop_data)
                expansion_info.node_to_representation[node_id] = node_attributes
            return node_attributes

        def sample_productions(expansion_info: ExpansionInformation, node_to_expand: int) -> List[Tuple[Iterable[str], int, float]]:
            prod_query_data = {}
            write_to_minibatch(prod_query_data,
                               self.placeholders['eg_production_node_representation'],
                               get_node_attributes(expansion_info, node_to_expand))
            if self.hyperparameters['eg_use_vars_for_production_choice']:
                vars_in_scope = list(expansion_info.variable_to_last_use_id.keys())
                vars_in_scope.remove(LAST_USED_TOKEN_NAME)
                vars_in_scope_representations = [get_node_attributes(expansion_info, expansion_info.variable_to_last_use_id[var])
                                                 for var in vars_in_scope]
                write_to_minibatch(prod_query_data,
                                   self.placeholders['eg_production_var_representations'],
                                   vars_in_scope_representations)
            if self.hyperparameters['eg_use_literal_copying'] or self.hyperparameters['eg_use_context_attention']:
                write_to_minibatch(prod_query_data,
                                   self.placeholders['context_token_representations'],
                                   expansion_info.context_token_representations)
                write_to_minibatch(prod_query_data,
                                   self.placeholders['context_token_mask'],
                                   expansion_info.context_token_mask)
            production_probs = self.sess.run(self.ops['eg_production_choice_probs'], feed_dict=prod_query_data)
            result = []
            # print("### Prod probs: %s" % (str(production_probs),))
            for picked_production_index in pick_indices_from_probs(production_probs, beam_size):
                prod_lhs, prod_rhs = production_id_to_production[picked_production_index]
                # TODO: This should be ensured by appropriate masking in the model
                if prod_lhs == expansion_info.node_to_type[node_to_expand]:
                    assert prod_lhs == expansion_info.node_to_type[node_to_expand]
                    result.append((prod_rhs, picked_production_index, production_probs[picked_production_index]))

            return result

        def sample_variable(expansion_info: ExpansionInformation, node_id: int) -> List[Tuple[str, float]]:
            vars_in_scope = list(expansion_info.variable_to_last_use_id.keys())
            vars_in_scope.remove(LAST_USED_TOKEN_NAME)
            vars_in_scope_representations = [get_node_attributes(expansion_info, expansion_info.variable_to_last_use_id[var])
                                             for var in vars_in_scope]
            var_query_data = {self.placeholders['eg_num_variable_choices']: len(vars_in_scope)}
            write_to_minibatch(var_query_data, self.placeholders['eg_varproduction_options_representations'], vars_in_scope_representations)
            # We choose the variable name based on the information of the /parent/ node:
            parent_node = expansion_info.node_to_parent[node_id]
            write_to_minibatch(var_query_data, self.placeholders['eg_varproduction_node_representation'], get_node_attributes(expansion_info, parent_node))
            var_probs = self.sess.run(self.ops['eg_varproduction_choice_probs'], feed_dict=var_query_data)

            result = []
            # print("### Var probs: %s" % (str(var_probs),))
            for picked_var_index in pick_indices_from_probs(var_probs, beam_size):
                result.append((vars_in_scope[picked_var_index], var_probs[picked_var_index]))
            return result

        def sample_literal(expansion_info: ExpansionInformation, node_id: int) -> List[Tuple[str, float]]:
            literal_kind_to_sample = expansion_info.node_to_type[node_id]
            lit_query_data = {}

            # We choose the literal based on the information of the /parent/ node:
            parent_node = expansion_info.node_to_parent[node_id]
            write_to_minibatch(lit_query_data, self.placeholders['eg_litproduction_node_representation'], get_node_attributes(expansion_info, parent_node))

            if self.hyperparameters["eg_use_literal_copying"]:
                write_to_minibatch(lit_query_data,
                                   self.placeholders['context_token_representations'],
                                   expansion_info.context_token_representations)
                write_to_minibatch(lit_query_data,
                                   self.placeholders['context_token_mask'],
                                   expansion_info.context_token_mask)
                write_to_minibatch(lit_query_data,
                                   self.placeholders['eg_litproduction_choice_normalizer'],
                                   expansion_info.literal_production_choice_normalizer[literal_kind_to_sample])
            lit_probs = self.sess.run(self.ops['eg_litproduction_choice_probs'][literal_kind_to_sample],
                                      feed_dict=lit_query_data)

            result = []
            # print("### Var probs: %s" % (str(lit_probs),))
            literal_vocab = self.metadata["eg_literal_vocabs"][literal_kind_to_sample]
            literal_vocab_size = len(literal_vocab)
            for picked_lit_index in pick_indices_from_probs(lit_probs, beam_size):
                if picked_lit_index < literal_vocab_size:
                    result.append((literal_vocab.id_to_token[picked_lit_index], lit_probs[picked_lit_index]))
                else:
                    result.append((expansion_info.context_tokens[picked_lit_index - literal_vocab_size], lit_probs[picked_lit_index]))
            return result

        def expand_node(expansion_info: ExpansionInformation) -> List[ExpansionInformation]:
            if len(expansion_info.nodes_to_expand) == 0:
                return [expansion_info]
            if expansion_info.num_expansions > max_decoding_steps:
                return []

            node_to_expand = expansion_info.nodes_to_expand.popleft()
            type_to_expand = expansion_info.node_to_type[node_to_expand]
            expansions = []

            if type_to_expand in self.metadata['eg_production_vocab']:
                # Case production from grammar
                for (prod_rhs, prod_id, prod_probability) in sample_productions(expansion_info, node_to_expand):
                    picked_rhs_expansion_info = clone_expansion_info(expansion_info, increment_expansion_counter=True)
                    picked_rhs_expansion_info.node_to_prod_id[node_to_expand] = prod_id
                    picked_rhs_expansion_info.expansion_logprob[0] = picked_rhs_expansion_info.expansion_logprob[0] + np.log(prod_probability)
                    # print("Expanding %i using rule %s -> %s with prob %.3f in %s (tree prob %.3f)."
                    #       % (node_to_expand, type_to_expand, prod_rhs, prod_probability,
                    #          " ".join(get_tokens_from_expansion(expansion_info, root_node)),
                    #          np.exp(expansion_info.expansion_logprob[0])))
                    # Declare all children
                    for child_node_type in prod_rhs:
                        child_node_id = declare_new_node(picked_rhs_expansion_info, node_to_expand, child_node_type)
                        # print("  Child %i (type %s)" % (child_node_id, child_node_type))
                    # Mark the children as expansions. As we do depth-first, push them to front of the queue; and as
                    # extendleft reverses the order and we do left-to-right, reverse that reversal:
                    picked_rhs_expansion_info.nodes_to_expand.extendleft(reversed(picked_rhs_expansion_info.node_to_children[node_to_expand]))
                    expansions.append(picked_rhs_expansion_info)
            elif type_to_expand == VARIABLE_NONTERMINAL:
                # Case choose variable name.
                if len(expansion_info.variable_to_last_use_id.keys()) > 1:  # Only continue if at least one var is in scope (not just LAST_USED_TOKEN_NAME)
                    for (child_label, var_probability) in sample_variable(expansion_info, node_to_expand):
                        # print("Expanding %i by using variable %s with prob %.3f in %s (tree prob %.3f)."
                        #       % (node_to_expand, child_label, var_probability,
                        #          " ".join(get_tokens_from_expansion(expansion_info, root_node)),
                        #          np.exp(expansion_info.expansion_logprob[0])))
                        child_expansion_info = clone_expansion_info(expansion_info)
                        child_expansion_info.node_to_synthesised_attr_node[node_to_expand] = node_to_expand  # synthesised and inherited are the same for leafs
                        child_expansion_info.node_to_label[node_to_expand] = child_label
                        self.compute_incoming_edges(self.metadata['eg_production_vocab'].keys(), child_expansion_info, node_to_expand)  # This needs to be done now before we update the variable-to-last-use info
                        child_expansion_info.expansion_logprob[0] = child_expansion_info.expansion_logprob[0] + np.log(var_probability)
                        if self.hyperparameters['eg_update_last_variable_use_representation']:
                            child_expansion_info.variable_to_last_use_id[child_label] = node_to_expand
                        child_expansion_info.variable_to_last_use_id[LAST_USED_TOKEN_NAME] = node_to_expand
                        expansions.append(child_expansion_info)
            elif type_to_expand in LITERAL_NONTERMINALS:
                for (picked_literal, literal_probability) in sample_literal(expansion_info, node_to_expand):
                    # print("Expanding %i by using literal %s with prob %.3f in %s (tree prob %.3f)."
                    #       % (node_to_expand, picked_literal, literal_probability,
                    #          " ".join(get_tokens_from_expansion(expansion_info, root_node)),
                    #          np.exp(expansion_info.expansion_logprob[0])))
                    picked_literal_expansion_info = clone_expansion_info(expansion_info)
                    picked_literal_expansion_info.node_to_synthesised_attr_node[node_to_expand] = node_to_expand  # synthesised and inherited are the same for leafs
                    picked_literal_expansion_info.node_to_label[node_to_expand] = picked_literal
                    self.compute_incoming_edges(self.metadata['eg_production_vocab'].keys(), picked_literal_expansion_info,
                                                node_to_expand)
                    picked_literal_expansion_info.expansion_logprob[0] = picked_literal_expansion_info.expansion_logprob[0] + np.log(literal_probability)
                    picked_literal_expansion_info.variable_to_last_use_id[LAST_USED_TOKEN_NAME] = node_to_expand
                    expansions.append(picked_literal_expansion_info)
            else:
                # Case node is a terminal: Do nothing
                # print("Handling leaf node %i (label %s)" % (node_to_expand, expansion_info.node_to_label[node_to_expand]))
                expansion_info.node_to_synthesised_attr_node[node_to_expand] = node_to_expand  # synthesised and inherited are the same for leafs
                self.compute_incoming_edges(self.metadata['eg_production_vocab'].keys(), expansion_info, node_to_expand)  # This needs to be done now before we update the variable-to-last-use info
                expansion_info.variable_to_last_use_id[LAST_USED_TOKEN_NAME] = node_to_expand
                expansions = [expansion_info]

            return expansions

        if self.hyperparameters['eg_use_literal_copying']:
            literal_production_choice_normalizer = {}
            for literal_kind in LITERAL_NONTERMINALS:
                # Collect all choices (vocab + things we can copy from):
                literal_vocab = self.metadata['eg_literal_vocabs'][literal_kind]
                literal_choices = literal_vocab.id_to_token + context_tokens
                first_tok_occurrences = {}
                num_choices = len(literal_vocab) + self.hyperparameters['eg_max_context_tokens']
                normalizer_map = np.arange(num_choices, dtype=np.int16)
                for (token_idx, token) in enumerate(literal_choices):
                    first_occ = first_tok_occurrences.get(token)
                    if first_occ is not None:
                        normalizer_map[token_idx] = first_occ
                    else:
                        first_tok_occurrences[token] = token_idx
                literal_production_choice_normalizer[literal_kind] = normalizer_map
        else:
            literal_production_choice_normalizer = None

        root_node = test_sample['eg_root_node']
        initial_variable_to_last_use_id = test_sample['eg_variable_eg_node_ids']
        initial_variable_to_last_use_id[LAST_USED_TOKEN_NAME] = test_sample['eg_last_token_eg_node_id']
        initial_node_to_representation = {node_id: initial_eg_node_representations[node_id]
                                          for node_id in initial_variable_to_last_use_id.values()}
        initial_node_to_representation[root_node] = initial_eg_node_representations[root_node]

        initial_info = ExpansionInformation(node_to_type={root_node: ROOT_NONTERMINAL},
                                            node_to_label={root_node: ROOT_NONTERMINAL},
                                            node_to_prod_id={},
                                            node_to_children=defaultdict(list),
                                            node_to_parent={},
                                            node_to_synthesised_attr_node={node_id: node_id for node_id in initial_node_to_representation.keys()},
                                            node_to_inherited_attr_node={},
                                            variable_to_last_use_id=initial_variable_to_last_use_id,
                                            node_to_representation=initial_node_to_representation,
                                            node_to_labeled_incoming_edges={root_node: defaultdict(list)},
                                            node_to_unlabeled_incoming_edges={root_node: defaultdict(list)},
                                            context_token_representations=context_token_representations,
                                            context_token_mask=context_token_mask,
                                            context_tokens=context_tokens,
                                            literal_production_choice_normalizer=literal_production_choice_normalizer,
                                            nodes_to_expand=deque([root_node]),
                                            expansion_logprob=[0.0],
                                            num_expansions=0)

        beams = [initial_info]
        while any(len(b.nodes_to_expand) > 0 for b in beams):
            new_beams = [new_beam
                         for beam in beams
                         for new_beam in expand_node(beam)]
            beams = sorted(new_beams, key=lambda b: -b.expansion_logprob[0])[:beam_size]  # Pick top K beams

        self.test_log("Groundtruth: %s" % (" ".join(test_sample['eg_tokens']),))

        all_predictions = []  # type: List[Tuple[List[str], float]]
        for (k, beam_info) in enumerate(beams):
            kth_result = get_tokens_from_expansion(beam_info, root_node)
            all_predictions.append((kth_result, np.exp(beam_info.expansion_logprob[0])))
            self.test_log("  @%i Prob. %.3f: %s" % (k+1, np.exp(beam_info.expansion_logprob[0]), " ".join(kth_result)))

        if len(beams) == 0:
            self.test_log("No beams finished!")

        return ModelTestResult(test_sample['eg_tokens'], all_predictions)