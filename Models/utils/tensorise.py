#!/usr/bin/env python
"""
Usage:
    tensorise.py [options] OUTPUT_BASE_FOLDER TYPE_LATTICE_FILE INPUT_FOLDER...

Tensorise a number of folders. Vocabularies and similar data is computed using
data from only the first folder (typically the training data fold).

Options:
    -h --help                  Show this screen.
    --max-num-files NUM        The maximum number of files to load [default: 100000]
    --hypers-override HYPERS   JSON dictionary overriding hyperparameter values.
    --model MODELNAME          Choose model type. [default: NAG]
    --metadata-to-use PATH     Select metadata to use. TYPE_LATTICE_FILE will be ignored.
    --for-test                 Flag indicating if the data should be tensorised as test data. [default: False]
    --debug                    Enable debug routines. [default: False]
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
"""
import json
import os
import sys
import traceback
import pdb

from docopt import docopt
from dpu_utils.utils import RichPath

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from exprsynth.model_restore_helper import get_model_class_from_name


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    type_lattice_file = RichPath.create(arguments['TYPE_LATTICE_FILE'], azure_info_path)
    output_folder = RichPath.create(arguments['OUTPUT_BASE_FOLDER'], azure_info_path)
    input_folders, input_folder_basenames = [], set()
    for input_folder_name in arguments['INPUT_FOLDER']:
        input_folder_basename = os.path.basename(input_folder_name)
        if input_folder_basename in input_folder_basenames:
            raise ValueError("Several input folders with same basename '%s'!" % (input_folder_basename,))
        input_folder_basenames.add(input_folder_basename)
        input_folder = RichPath.create(input_folder_name)
        assert input_folder.is_dir(), "%s is not a folder" % (input_folder,)
        input_folders.append(input_folder)

    model_class = get_model_class_from_name(arguments.get('--model', 'nag'))
    hyperparameters = model_class.get_default_hyperparameters()
    hypers_override = arguments.get('--hypers-override')
    if hypers_override is not None:
        hyperparameters.update(json.loads(hypers_override))

    model = model_class(hyperparameters, run_name=arguments.get('--run-name'))

    metadata_to_use = arguments.get('--metadata-to-use', None)
    if metadata_to_use is None:
        train_folder = input_folders[0]
        model.load_metadata(train_folder, type_lattice_file, max_num_files=int(arguments['--max-num-files']))
    else:
        metadata_path = RichPath.create(metadata_to_use, azure_info_path)
        model.load_existing_metadata(metadata_path)

    for_test = args.get('--for-test', False)
    model.make_model(is_train=not for_test)

    for input_folder in input_folders:
        input_folder_basename = input_folder.basename()
        this_output_folder = output_folder.join(input_folder_basename)
        this_output_folder.make_as_dir()
        model.tensorise_data_in_dir(input_folder,
                                    this_output_folder,
                                    for_test=for_test,
                                    max_num_files=int(arguments['--max-num-files']))


if __name__ == '__main__':
    args = docopt(__doc__)
    try:
        run(args)
    except:
        if args.get('--debug', False):
            _, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise
