import torch.utils.data as data
import glob
import os
import cv2
import numpy as np
from lib.utils.snake import snake_config
from lib.utils import data_utils
from lib.config import cfg
import tqdm
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
import numpy as np

from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config

from lib.utils.snake import snake_config, snake_cityscapes_utils, snake_eval_utils, snake_poly_utils
from external.cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
import pycocotools.mask as mask_util
import pycocotools.coco as coco

import json

mean = snake_config.mean
std = snake_config.std

def id_to_ano(class_ids):
    class_names = []
    for i in range (len(class_ids)):
        class_id = class_ids[i]
        if class_id == 1:
            class_name = 'pedestrian'
        elif class_id == 2:
            class_name = 'cyclist'
        elif class_id == 3:
            class_name = 'person-sitting'
        elif class_id == 4:
            class_name = 'car'
        elif class_id == 5:
            class_name = 'van'
        elif class_id == 6:
            class_name = 'tram'
        elif class_id == 7:
            class_name = 'truck'
        else: #class_id == 8:
            class_name = 'misc'
        class_names.append(class_name)
    return class_names


class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

        if os.path.isdir(cfg.demo_path):
            self.imgs = glob.glob(os.path.join(cfg.demo_path, '*'))
        elif os.path.exists(cfg.demo_path):
            self.imgs = [cfg.demo_path]
        else:
            raise Exception('NO SUCH FILE')

    def normalize_image(self, inp):
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - snake_config.mean) / snake_config.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        img = self.imgs[index]
        img = cv2.imread(img)

        width, height = img.shape[1], img.shape[0]
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = (int(width / 1.) | (x - 1)) + 1
        input_h = (int(height / 1.) | (x - 1)) + 1

        trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        inp = self.normalize_image(inp)
        ret = {'inp': inp}
        meta = {'center': center, 'scale': scale, 'test': '', 'ann': ''}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.imgs)


def demo():
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    dataset = Dataset()
    visualizer = make_visualizer(cfg)

    batch_index = 0
    file_names = []
    boxes = []
    scores = []
    polygons = []
    labels = []
    rles = []

    # Create JSON result structure
    result = {}
    res = []
    # Read elements from {lists of dicts}

    with open("data_file.json", "w") as write_file:
        for batch in tqdm.tqdm(dataset):
            batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()
            with torch.no_grad():
                output = network(batch['inp'], batch)
            visualizer.visualize(output, batch)

            file_names.append(dataset.imgs[batch_index])

            print(batch_index)
            inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
            box = output['detection'][:, :4].detach().cpu().numpy() * snake_config.down_ratio
            boxes.append(box)

            detection = output['detection']
            score = detection[:, 4].detach().cpu().numpy()
            label = detection[:, 5].detach().cpu().numpy().astype(int)
            py = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio

            scores.append(score)
            labels.append(id_to_ano(label))
            polygons.append(py)

            ori_h = inp.shape[0]
            ori_w = inp.shape[1]
            rle = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)
            rles.append(rle)

            result["Name"] = dataset.imgs[batch_index ]
            result["Object"] = id_to_ano(label)
            result["Score"] = score.tolist()
            result["rle"] = rle

            print("File name:", end=" ")
            print(result["Name"], end=" ")

            print("Object:", end=" ")
            print(result["Object"], end=" ")

            print("Score:", end=" ")
            print(result["Score"])

            print("rle:", end=" ")
            print(result["rle"])

            json.dump(result, write_file)
            #res.append(result)

            batch_index += 1
    print('Stop')

