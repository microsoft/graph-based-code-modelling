import pickle
from typing import Dict, Any, Optional, Type

import tensorflow as tf
from dpu_utils.utils import RichPath

from exprsynth.model import Model
from exprsynth.nagmodel import NAGModel
from exprsynth.graph2seqmodel import Graph2SeqModel
from exprsynth.seq2seqmodel import Seq2SeqModel
from exprsynth.seq2graphmodel import Seq2GraphModel


def get_model_class_from_name(model_name: str) -> Type[Model]:
    model_name = model_name.lower()
    if model_name in ['nag', 'nagmodel', 'graph2graph', 'graph2graphmodel']:
        return NAGModel
    elif model_name in ['graph2seq', 'graph2seqmodel']:
        return Graph2SeqModel
    elif model_name in ['seq2seq', 'seq2seqmodel']:
        return Seq2SeqModel
    elif model_name in ['seq2graph', 'seq2graphmodel']:
        return Seq2GraphModel
    else:
        raise Exception("Unknown model '%s'!" % model_name)


def restore(path: RichPath, is_train: bool, hyper_overrides: Optional[Dict[str, Any]]=None, model_save_dir: Optional[str]=None, log_save_dir: Optional[str]=None) -> Model:
    saved_data = path.read_by_file_suffix()

    if hyper_overrides is not None:
        saved_data['hyperparameters'].update(hyper_overrides)

    model_class = get_model_class_from_name(saved_data['model_type'])
    model = model_class(hyperparameters=saved_data['hyperparameters'],
                        run_name=saved_data.get('run_name'),
                        model_save_dir=model_save_dir,
                        log_save_dir=log_save_dir)
    model.metadata.update(saved_data['metadata'])
    model.make_model(is_train=is_train)

    variables_to_initialize = []
    with model.sess.graph.as_default():
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in sorted(model.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), key=lambda v: v.name):
                used_vars.add(variable.name)
                if variable.name in saved_data['weights']:
                    # print('Initializing %s from saved value.' % variable.name)
                    restore_ops.append(variable.assign(saved_data['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in sorted(saved_data['weights']):
                if var_name not in used_vars:
                    if var_name.endswith('Adam:0') or var_name.endswith('Adam_1:0') or var_name in ['beta1_power:0', 'beta2_power:0']:
                        continue
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            model.sess.run(restore_ops)
    return model
