import torch.nn as nn
from lib.utils import net_utils
import torch


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.act_crit = net_utils.FocalLoss()
        self.awh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.cp_crit = net_utils.FocalLoss()
        self.cp_wh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.ex_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit = torch.nn.functional.smooth_l1_loss

    def forward(self, batch):
        output = self.net(batch['inp'], batch)

        scalar_stats = {}
        loss = 0

        act_loss = self.act_crit(net_utils.sigmoid(output['act_hm']), batch['act_hm'])
        scalar_stats.update({'act_loss': act_loss})
        loss += act_loss

        awh_loss = self.awh_crit(output['awh'], batch['awh'], batch['act_ind'], batch['act_01'])
        scalar_stats.update({'awh_loss': awh_loss})
        loss += 0.1 * awh_loss

        act_01 = batch['act_01'].byte()

        cp_loss = self.cp_crit(net_utils.sigmoid(output['cp_hm']), batch['cp_hm'][act_01])
        scalar_stats.update({'cp_loss': cp_loss})
        loss += cp_loss

        cp_wh, cp_ind, cp_01 = [batch[k][act_01] for k in ['cp_wh', 'cp_ind', 'cp_01']]
        cp_wh_loss = self.cp_wh_crit(output['cp_wh'], cp_wh, cp_ind, cp_01)
        scalar_stats.update({'cp_wh_loss': cp_wh_loss})
        loss += 0.1 * cp_wh_loss

        # ex_loss = self.ex_crit(output['ex_pred'], output['i_gt_4py'])
        # scalar_stats.update({'ex_loss': ex_loss})
        # loss += ex_loss

        # py_loss = 0
        # output['py_pred'] = [output['py_pred'][-1]]
        # for i in range(len(output['py_pred'])):
        #     py_loss += self.py_crit(output['py_pred'][i], output['i_gt_py']) / len(output['py_pred'])
        # scalar_stats.update({'py_loss': py_loss})
        # loss += py_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

