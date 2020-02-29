import os
import imp


def make_visualizer(cfg):
    module = '.'.join(['lib.visualizers', cfg.task])
    path = os.path.join('lib/visualizers', cfg.task+'.py')
    visualizer = imp.load_source(module, path).Visualizer()
    return visualizer
