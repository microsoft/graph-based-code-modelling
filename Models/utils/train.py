#!/usr/bin/env python
"""
Usage:
    train.py [options] SAVE_FOLDER TRAIN_DATA_PATH VALID_DATA_PATH

Options:
    -h --help                  Show this screen.
    --max-num-epochs EPOCHS    The maximum number of epochs to run [default: 300]
    --hypers-override HYPERS   JSON dictionary overriding hyperparameter values.
    --model MODELNAME          Choose model type. [default: NAG]
    --run-name NAME            Picks a name for the trained model.
    --quiet                    Less output (not one per line per minibatch). [default: False]
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --debug                    Enable debug routines. [default: False]
"""
import json
import os
import sys
import time
from typing import Type, Dict, Any, Optional

from docopt import docopt
from dpu_utils.utils import RichPath, git_tag_run, run_and_debug


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from exprsynth.model_restore_helper import get_model_class_from_name
from exprsynth.model import Model
from exprsynth import model_restore_helper


def run_train(model_class: Type[Model],
              train_data_path: RichPath,
              valid_data_path: RichPath,
              save_folder: str,
              hyperparameters: Dict[str, Any],
              run_name: Optional[str]=None,
              quiet: bool=False) \
        -> RichPath:
    train_data_chunk_paths = train_data_path.get_filtered_files_in_dir('chunk_*')
    valid_data_chunk_paths = valid_data_path.get_filtered_files_in_dir('chunk_*')

    model = model_class(hyperparameters, run_name=run_name, model_save_dir=save_folder, log_save_dir=save_folder)
    if os.path.exists(model.model_save_path):
        model = model_restore_helper.restore(RichPath.create(model.model_save_path), is_train=True)
        model.train_log("Resuming training run %s of model %s with following hypers:\n%s" % (hyperparameters['run_id'],
                                                                                             model.__class__.__name__,
                                                                                             json.dumps(
                                                                                                 hyperparameters)))
        resume = True
    else:
        model.load_existing_metadata(train_data_path.join('metadata.pkl.gz'))
        model.make_model(is_train=True)
        model.train_log("Starting training run %s of model %s with following hypers:\n%s" % (hyperparameters['run_id'],
                                                                                             model.__class__.__name__,
                                                                                             json.dumps(hyperparameters)))
        resume = False
    model_path = model.train(train_data_chunk_paths, valid_data_chunk_paths, quiet=quiet, resume=resume)
    return model_path


def make_run_id(arguments: Dict[str, Any]) -> str:
    """Choose a run ID, based on the --save-name parameter and the current time."""
    user_save_name = arguments.get('--run-name')
    if user_save_name is not None:
        user_save_name = user_save_name[:-len('.pkl')] if user_save_name.endswith('.pkl') else user_save_name
        return "%s" % user_save_name
    else:
        user_save_name = arguments.get('--model', 'nag')
        return "%s-%s" % (user_save_name, time.strftime("%Y-%m-%d-%H-%M-%S"))


def run(arguments, tag_in_vcs=False) -> RichPath:
    azure_info_path = arguments.get('--azure-info', None)
    train_folder = RichPath.create(arguments['TRAIN_DATA_PATH'], azure_info_path)
    valid_folder = RichPath.create(arguments['VALID_DATA_PATH'], azure_info_path)
    save_folder = arguments['SAVE_FOLDER']

    assert train_folder.is_dir(), "%s is not a folder" % (train_folder,)
    assert valid_folder.is_dir(), "%s is not a folder" % (valid_folder,)

    model_class = get_model_class_from_name(arguments.get('--model', 'nag'))

    hyperparameters = model_class.get_default_hyperparameters()
    hypers_override = arguments.get('--hypers-override')
    if hypers_override is not None:
        hyperparameters.update(json.loads(hypers_override))
    hyperparameters['run_id'] = make_run_id(arguments)

    os.makedirs(save_folder, exist_ok=True)

    if tag_in_vcs:
        hyperparameters['git_commit'] = git_tag_run(hyperparameters['run_id'])

    return run_train(model_class, train_folder, valid_folder, save_folder, hyperparameters,
                     arguments.get('--run-name'), arguments.get('--quiet', False))


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
