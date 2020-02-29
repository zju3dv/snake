import torch.nn as nn
from lib.csrc.roi_align_layer.roi_align import ROIAlign
from lib.utils.rcnn_snake import rcnn_snake_config, rcnn_snake_utils
import torch
from lib.csrc.extreme_utils import _ext


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ComponentDetection(nn.Module):
    def __init__(self):
        super(ComponentDetection, self).__init__()

        self.pooler = ROIAlign((rcnn_snake_config.roi_h, rcnn_snake_config.roi_w))

        self.fusion = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.heads = {'cp_hm': 1, 'cp_wh': 2}
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                nn.Conv2d(256, classes, kernel_size=1, stride=1)
            )
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def prepare_training(self, cnn_feature, output, batch):
        w = cnn_feature.size(3)
        xs = (batch['act_ind'] % w).float()[..., None]
        ys = (batch['act_ind'] // w).float()[..., None]
        wh = batch['awh']
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        rois = rcnn_snake_utils.box_to_roi(bboxes, batch['act_01'].byte())
        roi = self.pooler(cnn_feature, rois)
        return roi

    def nms_class_box(self, box, score, cls, cls_num):
        box_score_cls = []

        for j in range(cls_num):
            ind = (cls == j).nonzero().view(-1)
            if len(ind) == 0:
                continue

            box_ = box[ind]
            score_ = score[ind]
            ind = _ext.nms(box_, score_, rcnn_snake_config.max_ct_overlap)

            box_ = box_[ind]
            score_ = score_[ind]

            ind = score_ > rcnn_snake_config.ct_score
            box_ = box_[ind]
            score_ = score_[ind]
            label_ = torch.full([len(box_)], j).to(box_.device).float()

            box_score_cls.append([box_, score_, label_])

        return box_score_cls

    def nms_abox(self, output):
        box = output['detection'][..., :4]
        score = output['detection'][..., 4]
        cls = output['detection'][..., 5]

        batch_size = box.size(0)
        cls_num = output['act_hm'].size(1)

        box_score_cls = []
        for i in range(batch_size):
            box_score_cls_ = self.nms_class_box(box[i], score[i], cls[i], cls_num)
            box_score_cls_ = [torch.cat(d, dim=0) for d in list(zip(*box_score_cls_))]
            box_score_cls.append(box_score_cls_)

        box, score, cls = list(zip(*box_score_cls))
        ind = torch.cat([torch.full([len(box[i])], i) for i in range(len(box))], dim=0)
        box = torch.cat(box, dim=0)
        score = torch.stack(score, dim=1)
        cls = torch.stack(cls, dim=1)

        detection = torch.cat([box, score, cls], dim=1)

        return detection, ind

    def prepare_testing(self, cnn_feature, output):
        if rcnn_snake_config.nms_ct:
            detection, ind = self.nms_abox(output)
        else:
            ind = output['detection'][..., 4] > rcnn_snake_config.ct_score
            detection = output['detection'][ind]
            ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(len(ind))], dim=0)

        ind = ind.to(cnn_feature.device)
        abox = detection[:, :4]
        roi = torch.cat([ind[:, None], abox], dim=1)

        roi = self.pooler(cnn_feature, roi)
        output.update({'detection': detection, 'roi_ind': ind})

        return roi

    def decode_cp_detection(self, cp_hm, cp_wh, output):
        abox = output['detection'][..., :4]
        adet = output['detection']
        ind = output['roi_ind']
        box, cp_ind = rcnn_snake_utils.decode_cp_detection(torch.sigmoid(cp_hm), cp_wh, abox, adet)
        output.update({'cp_box': box, 'cp_ind': cp_ind})

    def forward(self, output, cnn_feature, batch=None):
        z = {}

        if batch is not None and 'test' not in batch['meta']:
            roi = self.prepare_training(cnn_feature, output, batch)
            roi = self.fusion(roi)
            for head in self.heads:
                z[head] = self.__getattr__(head)(roi)

        if not self.training:
            with torch.no_grad():
                roi = self.prepare_testing(cnn_feature, output)
                roi = self.fusion(roi)
                cp_hm = self.cp_hm(roi)
                cp_wh = self.cp_wh(roi)
                self.decode_cp_detection(cp_hm, cp_wh, output)

        output.update(z)

        return output

