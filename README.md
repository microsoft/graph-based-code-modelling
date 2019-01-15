# Generative Code Modeling with Graphs

This is the code required to reproduce experiments in two of our papers on
modeling of programs, composed of three major components:
* A C# program required to extract (simplified) program graphs from C#
  source files, similar to our ICLR'18 paper
  [Learning to Represent Programs with Graphs](https://openreview.net/forum?id=BJOFETxR-).
  More precisely, it implements that paper apart from the speculative 
  dataflow component ("draw dataflow edges as if a variable would be used 
  in this place") and the alias analysis to filter equivalent variables.
* A TensorFlow model for program graphs, following ICLR'18 paper
  [Learning to Represent Programs with Graphs](https://openreview.net/forum?id=BJOFETxR-).
  This is a refactoring/partial rewrite of the original model, incorporating
  some new ideas on the representation of node labels from Cvitkovic et al.
  ([Open Vocabulary Learning on Source Code with a Graph-Structured Cache](https://arxiv.org/abs/1810.08305)).
* A TensorFlow model to generate new source code expressions conditional
  on their context, implementing our ICLR'19 paper
  [Generative Code Modeling with Graphs](https://openreview.net/forum?id=Bke4KsA5FX).

## Citations

If you want to cite this work for the encoder part (i.e., our ICLR'18 paper),
please use this bibtex entry:

```
@inproceedings{allamanis18learning,
  title={Learning to Represent Programs with Graphs},
  author={Allamanis, Miltiadis
          and Brockschmidt, Marc
          and Khademi, Mahmoud},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}
```

If you want to cite this work for the generative model (i.e., our ICLR'19
paper), please use this bibtex entry:

```
@inproceedings{brockschmidt2019generative,
  title={Generative Code Modeling with Graphs},
  author={Brockschmidt, Marc
          and Allamanis, Miltiadis
          and Gaunt, Alexander~L. 
          and Polozov, Oleksandr},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2019}
}
```

# Running the Code
The released code provides two components:
* Data Extraction: A C# project extracting graphs and expressions from a corpus
  of C# projects. The sources for this are in `DataExtraction/`.
* Modelling: A Python project learning model of expressions, conditionally on
  the program context. The sources for this are in `Models/`.

Note that the code is a research prototype; the documentation is generally
incomplete and code quality is varying.

## Data Extraction
### Building the data extractor
To build the data extraction, you need a .NET development environment (i.e.,
a working `dotnet` executable). Once this is set up, you can build the 
extractor as follows:
```
DataExtraction$ dotnet build
[...]
    ExpressionDataExtractor -> ExpressionDataExtractor\bin\Debug\net472\ExpressionDataExtractor.exe

Build succeeded.
[...]
```

### Using the data extractor
You can then use the resulting binary to extract contexts and expressions from
a C# project:
```
DataExtraction$ ExpressionDataExtractor/bin/Debug/net472/ExpressionDataExtractor.exe TestProject outputs/{graphs,types}
Writing graphs at outputs/graphs
Writing type hierarchies at outputs/types
[11/01/2019 14:07:05] Starting building all solutions in TestProject
[11/01/2019 14:07:05] Restoring packages for: TestProject/TinyTest.sln
[11/01/2019 14:07:05] In dir TestProject running nuget restore TinyTest.sln -NonInteractive -source https://api.nuget.org/v3/index.json
[11/01/2019 14:07:05] Nuget restore took 0 minutes.
[11/01/2019 14:07:06] Starting build of TestProject/TinyTest.sln
Compilations completed, completing extraction jobs...
Opening output file outputs/graphs/exprs-graph.0.jsonl.gz.
[11/01/2019 14:07:09] Extracted 15 expressions from TestProject/Program.cs.
```
Now, `outputs/graphs/exprs-graph.0.jsonl.gz` will contain (15) samples
consisting of a context graph and a target expression in tree form.
`ExpressionDataExtractor.exe --help` provides some information on
additional options.

*Note*: Building C# projects is often non-trivial (requiring [NuGet](https://www.nuget.org/)
and other libraries in the
path, preparing the build by running helper scripts, etc.). Roughly, data
extraction from a solution `Project.sln` will only succeed if running 
`MSBuild Project.sln` succeeds as well.

### Extractor Structure
Data extraction is split into two projects:
* `ExpressionDataExtractor`: This is the actual command-line utility and with
  some code to find and build C# projects in a directory tree.
* `SourceGraphExtractionUtils`: This project contains the actual extraction
  logic. Almost all of the interesting logic is in `GraphDataExtractor`, which
  is in dire need of a refactoring. This class does four complex things:
    - Identify target expressions to extract (`SimpleExpressionIdentifier`).
    - Turn expressions into a simplified version of the C# syntax tree
      (`TurnIntoProductions`). This is needed because Roslyn does not expose an
      /abstract/ syntax tree, but a /full/ ST with all surface code artifacts.
    - Construction of a Program Graph as in "Learning to Represent Programs
      with Graphs", ICLR'18 (`ExtractSourceGraph`).
    - Extraction of a subgraph of limited size around a target expression,
      removing the target expression in the process (`CopySubgraphAroundHole`).
  
  There is some bare-bones documentation for these components, but if you
  are trying to understand them and are stuck, open an issue with concrete
  questions and better documentation will magically appear.

## Models
First, run `pip install -r requirements.txt` to download the needed
dependencies. Note that all code is written in Python 3.

As the preprocessing of graphs into tensorised form is relatively computationally expensive,
we use a preprocessing step to do this. This computes vocabularies, the
grammar required to produce the observed expressions and so on, and then
transforms node labels from string form into tensorised form, etc.:
```
$ utils/tensorise.py test_data/tensorised test_data/exprs-types.json.gz test_data/graphs/
Imputed grammar:
  Expression  -[00]->  ! Expression
  Expression  -[01]->  - Expression
  Expression  -[02]->  -- Expression
  Expression  -[03]->  CharLiteral
  Expression  -[04]->  Expression * Expression
  Expression  -[05]->  Expression + Expression
  Expression  -[06]->  Expression ++
  Expression  -[07]->  Expression . IndexOf ( Expression )
  Expression  -[08]->  Expression . IndexOf ( Expression , Expression , Expression )
  Expression  -[09]->  Expression . StartsWith ( Expression )
  Expression  -[10]->  Expression < Expression
  Expression  -[11]->  Expression > Expression
  Expression  -[12]->  Expression ? Expression : Expression
  Expression  -[13]->  Expression [ Expression ]
  Expression  -[14]->  IntLiteral
  Expression  -[15]->  StringLiteral
  Expression  -[16]->  Variable
Known literals:
  IntLiteral: ['%UNK%', '0', '1', '2', '4', '43']
  CharLiteral: ['%UNK%', "'-'"]
  StringLiteral: ['"foobar"', '%UNK%']
Tensorised 15 (15 before filtering) samples from 'test_data/graphs/' into 'test_data/tensorised/'.
```

If you want to use a given vocabulary/grammar (e.g., to prepare validation
data), you can use the computed metadata from another folder:
```
$ utils/tensorise.py --metadata-to-use test_data/tensorised/metadata.pkl.gz test_data/tensorised_valid test_data/exprs-types.json.gz test_data/graphs/
Tensorised 15 (15 before filtering) samples from 'test_data/graphs/' into 'test_data/tensorised_valid/'.
```

### Training
To test if everything works, training on a small number of examples
should work:
```
% utils/train.py trained_models/overtrain test_data/tensorised/{,}
Starting training run NAG-2019-01-11-18-21-14 of model NAGModel with following hypers:
{"optimizer": "Adam", "seed": 0, "dropout_keep_rate": 0.9, "learning_rate": 0.00025, "learning_rate_decay": 0.98, "momentum": 0.85, "gradient_clip": 1, "max_epochs": 100, "patience": 5, "max_num_cg_nodes_in_batch": 100000, "excluded_cg_edge_types": [], "cg_add_subtoken_nodes": true, "cg_node_label_embedding_style": "Token", "cg_node_label_vocab_size": 10000, "cg_node_label_char_length": 16, "cg_node_label_embedding_size": 32, "cg_node_type_vocab_size": 54, "cg_node_type_max_num": 10, "cg_node_type_embedding_size": 32, "cg_ggnn_layer_timesteps": [3, 1, 3, 1], "cg_ggnn_residual_connections": {"1": [0], "3": [0, 1]}, "cg_ggnn_hidden_size": 64, "cg_ggnn_use_edge_bias": false, "cg_ggnn_use_edge_msg_avg_aggregation": false, "cg_ggnn_use_propagation_attention": false, "cg_ggnn_graph_rnn_activation": "tanh", "cg_ggnn_graph_rnn_cell": "GRU", "eg_token_vocab_size": 100, "eg_literal_vocab_size": 10, "eg_max_variable_choices": 10, "eg_propagation_substeps": 50, "eg_hidden_size": 64, "eg_edge_label_size": 16, "exclude_edge_types": [], "eg_graph_rnn_cell": "GRU", "eg_graph_rnn_activation": "tanh", "eg_use_edge_bias": false, "eg_use_vars_for_production_choice": true, "eg_update_last_variable_use_representation": true, "eg_use_literal_copying": true, "eg_use_context_attention": true, "eg_max_context_tokens": 500, "run_id": "NAG-2019-01-11-18-21-14"}
==== Epoch 0 ====
  Epoch 0 (train) took 0.26s [processed 58 samples/second]
 Training Loss: 10.622053
  Epoch 0 (valid) took 0.11s [processed 132 samples/second]
 Validation Loss: 9.516558
  Best result so far -- saving model as 'trained_models/overtrain/NAGModel_NAG-2019-01-11-18-21-14_model_best.pkl.gz'.
[...]
==== Epoch 100 ====
  Epoch 100 (train) took 0.22s [processed 69 samples/second]
 Training Loss: 0.650332
  Epoch 100 (valid) took 0.10s [processed 146 samples/second]
 Validation Loss: 0.637806
```

We can then evaluate the model:
```
$ utils/test.py trained_models/overtrain/NAGModel_NAG-2019-01-11-18-21-14_model_best.pkl.gz test_data/graphs/ trained_models/overtrain/test_results/
[...]
Groundtruth: b ? 1 : - i
  @1 Prob. 0.219: b ? 1 : i
  @2 Prob. 0.095: b ? 1 : - i
  @3 Prob. 0.066: b ++
  @4 Prob. 0.041: b ? 2 : i
  @5 Prob. 0.040: b ? 1 : arr
Num samples: 15 (15 before filtering)
Avg Sample Perplexity: 1.51
Std Sample Perplexity: 0.27
Accuracy@1: 60.0000%
Accuracy@5: 86.6667%
```

### Model Variations
There are four different model types implemented:
 * `NAG`: The main model presented in "Generative Code Modeling with Graphs" 
   (ICLR'19), representing the program context by a graph and using the
   graph-structured decoding strategy discussed in the paper.
 * `seq2graph`: An ablation that uses the graph-structured decoder, but
   represents the context using a sequence model. Concretely, a window
   of tokens around the hole to fill is fed into a two-layer BiGRU to
   obtain a representation for the program context.
   Additionally, for each variable in scope, a number of token windows
   around usages are encoded with a second BiGRU, and their
   representation is averaged.
* `graph2seq`: An ablation that uses a graph to represent the program context,
  but then relies on a 2-layer GRU to construct the target expression.
* `seq2seq`: An ablation using both a sequence encoder for the program context
  as well as a sequence decoder.
All models have a wide range of different hyperparameters.

As these choices influence the format of tensorised data, both `tensorise.py`
and `train.py` need to be re-run for every variation:
```
$ utils/tensorise.py test_data/tensorised_seq2graph test_data/exprs-types.json.gz test_data/graphs/ --model seq2graph --hypers-override '{"eg_hidden_size": 32, "cx_token_representation_size": 64}'
[...]
$ utils/train.py test_data/tensorised_seq2graph/{,} --model seq2graph --hypers-override '{"eg_hidden_size": 32, "cx_token_representation_size": 64}'
[...]
```

### Model Structure

Roughly, the model code is split into three main components:
* Infrastructure: The `Model` class (in `exprsynth/model.py`) implements the
  usual general infrastructure bits; and models are expected to implement
  certain hooks in it. All of these are documented individually.
  - Saving and loading models, hyperparameters, training loop, etc.
  - Construction of metadata such as vocabularies: This code is parallelised
    and implementations need to extend three core methods (`_init_metadata`,
    `_load_metadata_from_sample`, `_finalise_metadata`) to use this code.
    Intuitively, `init_metadata` prepares a `dict` to store raw information
    (e.g., token counters) and `_load_metadata_from_sample` processes a 
    single datapoint to update this raw data. These two are usually
    parallelised, and `_finalise_metadata` has to combine all raw metadata
    dictionaries to obtain one metadata dictionary, containing for example
    a vocabulary (in a MapReduce style).
  - Tensorising raw samples: `_load_data_from_sample` needs to be extended
    for this, and implementors can use the computed metadata.
  - Minibatch construction: We build minibatches by growing a batch until
    we reach a size limit (e.g., because we hit the maximal number of nodes
    per batch). This is implemented in the methods `_init_minibatch`
    (creating a new dictionary to hold data), `_extend_minibatch_by_sample`
    to add a new sample to a batch, and `_finalise_minibatch`, which can do
    final flattening operations and turn things into a feed dict.
    
    *Note* 1: This somewhat complicated strategy is required for two
    reasons. First, the sizes of graphs can vary substantially, and so
    picking a fixed number of graphs may yield a minibatch that is very
    small or large (in number of nodes). At the same time, our strategy
    of treating graph batches as one large graph requires regular shifting
    of node indices of samples, which is easiest to implement correctly
    in this incremental fashion.

    *Note* 2: In principle, these methods should be executed on another
    thread, so that a new minibatch can be constructed while the GPU is
    computing. Code for this exists, but was taken out for simplicity here.
 * Context Models: Two context models are implemented:
   - `ContextGraphModel` (in `exprsynth/contegraphmodel.py`): This is
     the code implementing the modeling of a program graph, taking types,
     labels, dataflow, etc. into account. It produces a representation of
     all nodes in the input graphs as `model.ops['cg_node_representations']`,
     which can then be used in downstream models.
   - `ContextTokenModel` (in `exprsynth/contexttokenmodel.py`): This implements
     a simple BiGRU model over the tokens around the target expression, taking
     types into account. It produces a representation of all tokens in these
     contexts as `model.ops['cx_hole_context_representations']`.
 * Decoder Models: Two decoder models are implemented:
   - `NAGDecoder` (in `exprsynth/nagdecoder.py`): This is the code implementing
     the modeling of program generation as a graph.

     First, a representation of all nodes in the expansion graph is computed
     using scheduled message passing (implemented using `AsyncGGNN` at training
     time and step-wise use of `get_node_attributes` at test time).
     The schedule is determined by the `__load_expansiongraph_training_data_from_sample`
     method (and is the core of our paper).

     Second, a number of expansion decisions are made. The modeling of grammar
     productions is in `__make_production_choice_logits_model`, variables are
     chosen in `__make_variable_choice_logits_model` and literals are produced
     or copied in `__make_literal_choice_logits_model`.
   - `SeqDecoder` (in `exprsynth/seqdecoder.py`): A simple sequence decoder.
 * Glue code: Context models and decoders are combined using the actual models
   we instantiate. For example, `NAGModel` extends the `ContextGraphModel` and
   instantiates a `NAGDecoder`, contributing only some functionality to forward
   data from the encoder to the decoder.


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
