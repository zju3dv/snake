import torch.nn as nn
from .dla import DLASeg
from lib.utils import net_utils
from .cp_head import ComponentDetection
import torch
from lib.utils.snake import snake_decode
from lib.utils import data_utils


class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv)
        self.cp = ComponentDetection()

    def decode_detection(self, output, h, w):
        ct_hm = output['act_hm']
        wh = output['awh']
        ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return ct, detection

    def forward(self, x, batch=None):
        output, cnn_feature = self.dla(x)
        with torch.no_grad():
            self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))
        output = self.cp(output, cnn_feature, batch)
        return output


def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network

