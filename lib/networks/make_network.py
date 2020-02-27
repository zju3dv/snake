import os
import imp


_network_factory = {
}


def get_network(cfg):
    arch = cfg.network
    heads = cfg.heads
    head_conv = cfg.head_conv
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _network_factory[arch]
    network = get_model(num_layers, heads, head_conv)
    return network


def make_network(cfg):
    module = '.'.join(['lib.networks', cfg.task])
    path = os.path.join('lib/networks', cfg.task, '__init__.py')
    return imp.load_source(module, path).get_network(cfg)
