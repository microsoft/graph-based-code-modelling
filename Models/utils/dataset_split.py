#!/usr/bin/env python
"""
Usage:
    graphsplit.py [options] ALL_GRAPHS_FOLDER OUTPUT_FOLDER

Options:
    -h --help                    Show this screen.
    --train-ratio FLOAT          Ratio of files for training set. [default: 0.6]
    --valid-ratio FLOAT          Ratio of files for validation set. [default: 0.2]
    --test-ratio FLOAT           Ratio of files for test set. [default: 0.2]
    --test-only-projects=<File>  A file containing the project names of the test-only data.
    --azure-info=<path>          Azure authentication information file (JSON). Used to load data from Azure storage.
    --seed INT                   Random seed. [default: 0]
"""
import hashlib
from docopt import docopt
from typing import Dict, Set
from multiprocessing import Pool

import numpy as np
from dpu_utils.utils import RichPath


def get_fold(filename: str, train_ratio: float, valid_ratio: float, test_only_projects: Set[str]) -> str:
    # Find the name of the project, which is separated by \\.
    # Note that a canonical name starts with \\, hence the [1:]
    first_slash_idx = filename[1:].find('\\')
    if first_slash_idx == -1:
        print('Error in finding project name of %s' % filename)
    else:
        project_name = filename[1:first_slash_idx+1]
        if project_name in test_only_projects:
            return 'test-only'

    hash_val = int(hashlib.md5(filename.encode()).hexdigest(), 16) % (2**16)
    train_bound = int(2**16 * train_ratio)
    if hash_val <= train_bound:
        return "train"
    elif hash_val <= train_bound + int(2**16 * valid_ratio):
        return "valid"
    else:
        return "test"


def split_file(input_path: RichPath, output_paths: Dict[str, RichPath], train_ratio: float,
               valid_ratio: float, test_ratio: float, test_only_projects: Set[str]) -> None:
    train_graphs, valid_graphs, test_graphs, test_only_graphs = [], [], [], []

    try:
        for datapoint in input_path.read_by_file_suffix():
            datapoint_provenance = datapoint['Filename']
            file_set = get_fold(datapoint_provenance, train_ratio, valid_ratio, test_only_projects)
            if file_set == 'train':
                train_graphs.append(datapoint)
            elif file_set == 'valid':
                valid_graphs.append(datapoint)
            elif file_set == 'test':
                test_graphs.append(datapoint)
            elif file_set == 'test-only':
                test_only_graphs.append(datapoint)
    except EOFError:
        print('Failed for file %s.' % input_path)
        return

    input_file_basename = input_path.basename()

    if train_ratio > 0:
        output_path = output_paths['train'].join(input_file_basename)
        print('Saving %s...' % (output_path,))
        output_path.save_as_compressed_file(train_graphs)

    if valid_ratio > 0:
        output_path = output_paths['valid'].join(input_file_basename)
        print('Saving %s...' % (output_path,))
        output_path.save_as_compressed_file(valid_graphs)

    if test_ratio > 0:
        output_path = output_paths['test'].join(input_file_basename)
        print('Saving %s...' % (output_path,))
        output_path.save_as_compressed_file(test_graphs)

    if len(test_only_graphs) > 0:
        output_path = output_paths['test-only'].join(input_file_basename)
        print('Saving %s...' % (output_path,))
        output_path.save_as_compressed_file(test_only_graphs)


def split_many_files(input_dir: RichPath, output_dir: RichPath, train_ratio: float, valid_ratio: float,
                     test_ratio: float, test_only_projects: Set[str]) -> None:
    output_paths = {}  # type: Dict[str, RichPath]
    for split_name in ['train', 'valid', 'test', 'test-only']:
        graph_dir_name_for_split_type = input_dir.basename() + '-' + split_name
        graph_dir_for_split_type = output_dir.join(graph_dir_name_for_split_type)
        output_paths[split_name] = graph_dir_for_split_type
        graph_dir_for_split_type.make_as_dir()

    pool = Pool()
    pool.starmap(split_file,
                 [(f, output_paths, train_ratio, valid_ratio, test_ratio, test_only_projects)
                  for f in input_dir.get_filtered_files_in_dir('*')])

    return None


if __name__ == '__main__':
    args = docopt(__doc__)
    train_ratio = float(args['--train-ratio'])
    valid_ratio = float(args['--valid-ratio'])
    test_ratio = float(args['--test-ratio'])
    test_only_projects = set()  # type: Set[str]
    if args.get('--test-only-projects') is not None:
        with open(args.get('--test-only-projects')) as f:
            for line in f:
                if len(line.strip()) > 0:
                    test_only_projects.add(line.strip())
    assert (train_ratio + valid_ratio + test_ratio <= 1)

    np.random.seed(int(args['--seed']))

    azure_info_path = args.get('--azure-info', None)
    graphs_folder = args['ALL_GRAPHS_FOLDER']
    if graphs_folder.endswith('/'):  # Split off final separator so that succeeding basename()s are not returning an empty string...
        graphs_folder = graphs_folder[:-1]
    graphs_folder = RichPath.create(graphs_folder, azure_info_path)
    output_folder = RichPath.create(args['OUTPUT_FOLDER'], azure_info_path)

    split_many_files(graphs_folder, output_folder, train_ratio, valid_ratio, test_ratio, test_only_projects)

    print('Splitting finished.')
