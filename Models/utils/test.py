#!/usr/bin/env python
"""
Usage:
    test.py [options] MODEL_PATH TEST_DATA_PATH OUTPUT_FOLDER

Options:
    -h --help                  Show this screen.
    --num-processes NUMBER     Number of parallel processes to use for testing. [default: 1]
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
"""
import os
import sys
import time
from typing import Any, Dict, List, Tuple
from multiprocessing import Pool

import numpy as np
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from dpu_utils.mlutils.vocabulary import Vocabulary

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from exprsynth import model_restore_helper


def chunkify(lst: List[Any], num_chunks: int) -> List[List[Any]]:
    return [lst[i::num_chunks] for i in range(num_chunks)]


def build_csharp_check_function(raw_sample: Dict[str, Any], expression: str) -> str:
    sample_types = raw_sample['ContextGraph']['NodeTypes']
    return_type = sample_types['0'] if sample_types['0'] != '?' else 'object'
    in_scope_var_types = [(var_name, sample_types[str(var_node_id)])
                          for (var_name, var_node_id) in raw_sample['LastUseOfVariablesInScope'].items()]
    test_func_signature = "static %s Test(%s)" % (return_type,
                                                  ", ".join("%s %s" % (var_type, var_name)
                                                            for (var_name, var_type) in in_scope_var_types))
    expression = expression.replace('%UNK%', 'unkVariableName').replace('<String Literal>', '"UNK String Literal"')\
                           .replace('<Numeric Literal>', '12345').replace('<Char Literal>', "'?'")
    test_func = "%s { return %s; }" % (test_func_signature, expression)
    return "//%s:%s\n\nclass TestClass {\n%s\n}\n" % (raw_sample['Filename'], raw_sample['HoleLineSpan'], test_func)


def token_seq_equal(a: List[str], b: List[str]):
    unk_tok = Vocabulary.get_unk()
    if unk_tok in a or unk_tok in b:
        return False
    else:
        return a == b


def test_on_raw_chunks(model_path: RichPath,
                       test_hyper_overrides: Dict[str, Any],
                       snippet_output_folder: str,
                       proc_id: int,
                       test_raw_data_chunks: List[RichPath]) -> Tuple[int, List[float], int, int]:
    def write_snippet(snippet_id: int, content: str):
        with open(os.path.join(snippet_output_folder, 'sample_%i-%i.cs' % (proc_id, snippet_id)), 'w', encoding='utf-8') as f:
            f.write(content)

    results = {"correct_at_1": 0,
               "correct_at_5": 0,
               "token_perplexities": []}

    def per_result_callback(sample_idx, token_perplexity, raw_sample, sample_result):
        predictions = sample_result.all_predictions
        results["token_perplexities"].append(token_perplexity)
        if len(predictions) == 0:
            write_snippet(sample_idx, build_csharp_check_function(raw_sample, '???'))  # A full error
            return
        if token_seq_equal(predictions[0][0], sample_result.ground_truth):
            results["correct_at_1"] += 1
        if any(token_seq_equal(prediction[0], sample_result.ground_truth) for prediction in predictions[:5]):
            results["correct_at_5"] += 1
        write_snippet(sample_idx, build_csharp_check_function(raw_sample, ' '.join(predictions[0][0])))

    test_hyper_overrides['run_id'] = test_hyper_overrides['run_id'] + "-" + str(proc_id)
    test_hyper_overrides['gpu_device_id'] = ""
    train_model = model_restore_helper.restore(model_path, is_train=True, hyper_overrides=test_hyper_overrides)
    model = model_restore_helper.restore(model_path, is_train=False, hyper_overrides=test_hyper_overrides)
    num_samples = model.test(test_raw_data_chunks, per_result_callback_fn=per_result_callback, train_model=train_model)
    return num_samples, results["token_perplexities"], results["correct_at_1"], results["correct_at_5"]


def run_test(model_path: RichPath, test_data_path: RichPath, output_folder: str, num_processes: int):
    test_run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])

    test_hyper_overrides = {
                             "run_id": test_run_id,
                             "cx_max_num_types": 50,
                             "cg_max_num_types": 50,
                             "eg_propagation_substeps": 100,
                             "eg_max_variable_choices": 15,
                             "dropout_keep_rate": 1.0,
                            }

    test_data_chunks = test_data_path.get_filtered_files_in_dir('*gz')

    test_jobs = [(model_path, test_hyper_overrides, output_folder, chunk_id, chunk_data_paths)
                 for chunk_id, chunk_data_paths in enumerate(chunkify(test_data_chunks, num_processes))]
    with Pool(processes=num_processes) as pool:
        num_samples, token_perplexities, correct_at_1, correct_at_5 = zip(*pool.starmap(test_on_raw_chunks, test_jobs))
    # num_samples, token_perplexities, correct_at_1, correct_at_5 = zip(*[test_on_raw_chunks(*job) for job in test_jobs])

    num_samples = sum(num_samples)
    token_perplexities = np.concatenate(token_perplexities, axis=0)
    correct_at_1 = sum(correct_at_1)
    correct_at_5 = sum(correct_at_5)

    print('Num samples: %i (%i before filtering)' % (len(token_perplexities), num_samples))
    print('Avg Sample Perplexity: %.2f' % np.mean(token_perplexities))
    print('Std Sample Perplexity: %.2f' % np.std(token_perplexities))
    print('Accuracy@1: %.4f%%' % (float(correct_at_1) / num_samples * 100))
    print('Accuracy@5: %.4f%%' % (float(correct_at_5) / num_samples * 100))


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    test_folder = RichPath.create(arguments['TEST_DATA_PATH'], azure_info_path)
    model_path = RichPath.create(arguments['MODEL_PATH'])
    output_folder = arguments['OUTPUT_FOLDER']
    os.makedirs(output_folder, exist_ok=True)
    num_processes = int(arguments['--num-processes'])
    run_test(model_path, test_folder, output_folder, num_processes)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
