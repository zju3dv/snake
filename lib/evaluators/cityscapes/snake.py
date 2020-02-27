import os
import cv2
from lib.utils.snake import snake_config, snake_cityscapes_utils, snake_eval_utils, snake_poly_utils
from external.cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
from lib.utils import data_utils


class Evaluator:
    def __init__(self, result_dir):
        self.anns = []

        self.result_dir = result_dir
        self.instance_dir = os.path.join(result_dir, 'mask')
        self.txt_dir = os.path.join(result_dir, 'text')

        os.system('mkdir -p {}'.format(self.instance_dir))
        os.system('mkdir -p {}'.format(self.txt_dir))

    def evaluate(self, output, batch):
        detection = output['detection']
        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)
        label = snake_cityscapes_utils.continuous_label_to_cityscapes_label(label)
        py = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        ori_h, ori_w = 1024, 2048
        mask = snake_eval_utils.poly_to_mask(py, label, ori_h, ori_w)

        img_id = batch['meta']['img_id'][0]
        instance_dir = os.path.join(self.instance_dir, img_id)
        os.system('mkdir -p {}'.format(instance_dir))

        self.anns.append(batch['meta']['ann'][0])
        txt_path = os.path.join(self.txt_dir, '{}.txt'.format(img_id))
        with open(txt_path, 'w') as f:
            for i in range(len(label)):
                instance_path = os.path.join(instance_dir, 'instance'+str(i)+'.png')
                cv2.imwrite(instance_path, mask[i])
                instance_path = os.path.join('../mask', img_id, 'instance'+str(i)+'.png')
                f.write('{} {} {}\n'.format(instance_path, label[i], score[i]))

    def summarize(self):
        prediction = []
        gt = []
        for ann in self.anns:
            split, city, file_name = ann.split('/')[-3:]
            img_id = file_name.replace('.json', '')
            prediction.append(os.path.join(self.txt_dir, img_id+'.txt'))
            gt.append(os.path.join('data/cityscapes/gtFine', split, city, img_id+'_gtFine_instanceIds.png'))
        self.anns = []
        ap = evalInstanceLevelSemanticLabeling.evaluate(prediction, gt, self.result_dir)
        return {'ap': ap}

