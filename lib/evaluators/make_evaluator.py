import imp
import os
from lib.datasets.dataset_catalog import DatasetCatalog


def _evaluator_factory(cfg):
    task = cfg.task
    data_source = DatasetCatalog.get(cfg.test.dataset)['id']
    module = '.'.join(['lib.evaluators', data_source, task])
    path = os.path.join('lib/evaluators', data_source, task+'.py')
    evaluator = imp.load_source(module, path).Evaluator(cfg.result_dir)
    return evaluator


def make_evaluator(cfg):
    if cfg.skip_eval:
        return None
    else:
        return _evaluator_factory(cfg)
