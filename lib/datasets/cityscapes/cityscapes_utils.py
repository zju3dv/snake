import numpy as np
import glob
import os
import json

# Globals ----------------------------------------------------------------------
COCO_LABELS = {24: 1,
               26: 2,
               27: 3,
               25: 4,
               33: 5,
               32: 6,
               28: 7,
               31: 8}

# Label number to name and color
INSTANCE_LABELS = {26: {'name': 'car', 'color': [0, 0, 142]},
                   24: {'name': 'person', 'color': [220, 20, 60]},
                   25: {'name': 'rider', 'color': [255, 0, 0]},
                   32: {'name': 'motorcycle', 'color': [0, 0, 230]},
                   33: {'name': 'bicycle', 'color': [119, 11, 32]},
                   27: {'name': 'truck', 'color': [0, 0, 70]},
                   28: {'name': 'bus', 'color': [0, 60, 100]},
                   31: {'name': 'train', 'color': [0, 80, 100]}}

# Label name to number
LABEL_DICT = {'car': 26, 'person': 24, 'rider': 25, 'motorcycle': 32,
              'bicycle': 33, 'truck': 27, 'bus': 28, 'train': 31}
# LABEL_DICT = {'bicycle': 26}

# Label name to contiguous number
JSON_DICT = dict(car=0, person=1, rider=2, motorcycle=3, bicycle=4, truck=5, bus=6, train=7)
# JSON_DICT = dict(bicycle=0)
# Contiguous number to name
NUMBER_DICT = {0: 'car', 1: 'person', 2: 'rider', 3: 'motorcycle',
             4: 'bicycle', 5: 'truck', 6: 'bus', 7: 'train'}
# NUMBER_DICT = {0:'bicycle'}
# Array of keys
KEYS = np.array([[26000, 26999], [24000, 24999], [25000, 25999],
                 [32000, 32999], [33000, 33999], [27000, 27999],
                 [28000, 28999], [31000, 31999]])
# ------------------------------------------------------------------------------


def visualize_bbox(img, boxes):
    """
    img: [h, w, 3]
    boxes: [n, 4], [[x, y, x_max, y_max]]
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    _, ax = plt.subplots(1)
    ax.imshow(img)
    n = len(boxes)
    for ni in range(n):
        x, y, x_max, y_max = boxes[ni]
        ax.add_patch(patches.Polygon(xy=[[x, y], [x, y_max], [x_max, y_max], [x_max, y]], fill=False, linewidth=1, edgecolor='r'))
    plt.show()


def add_stat(stat_dict, examples, size, img_stat):
    flag_s = False
    flag_m = False
    flag_l = False
    for obj in examples:
        if not size:
            stat_dict[obj['label']] += 1
        elif img_stat:
            area = 0
            for per in obj['components']:
                area += per['area']
            if area < 32 * 32 and not flag_s:
                flag_s = True
                stat_dict['small'] += 1
            elif area < 96 * 96 and not flag_m:
                flag_m = True
                stat_dict['medium'] += 1
            elif not flag_l:
                flag_l = True
                stat_dict['large'] += 1
        else:
            area = 0
            for per in obj['components']:
                area += per['area']
            if area < 32 * 32:
                stat_dict['small'] += 1
            elif area < 96 * 96:
                stat_dict['medium'] += 1
            else:
                stat_dict['large'] += 1
    return stat_dict


def read_dataset(ann_files, size=False, img_stat=False):
    if not isinstance(ann_files, tuple):
        ann_files = (ann_files,)

    ann_file = []
    for ann_file_dir in ann_files:
        ann_file += glob.glob(os.path.join(ann_file_dir, '*/*.json'))
    if not size:
        stat_dict = dict(person=0, rider=0, car=0, truck=0, bus=0, train=0, motorcycle=0, bicycle=0)
    else:
        stat_dict = dict(small=0, medium=0, large=0)

    ann_filter = []
    for fname in ann_file:
        with open(fname, 'r') as f:
            ann = json.load(f)
            examples = []
            for instance in ann:
                instance_label = instance['label']
                if instance_label not in LABEL_DICT:
                    continue
                examples.append(instance)
            if len(examples) > 0:
                ann_filter.append(fname)
                stat_dict = add_stat(stat_dict, examples, size, img_stat)

    return stat_dict


def visualize_stat_num():
    ann_files = ('data/cityscapes/annotations/train', 'data/cityscapes/annotations/train_val')
    stat = read_dataset(ann_files, True, False)
    name = list(stat.keys())
    num = list(stat.values())
    import matplotlib.pyplot as plt
    plt.bar(range(len(num)), num, color='b', tick_label=name)
    for a, b in zip(range(len(num)), num):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    plt.show()
    import ipdb; ipdb.set_trace()
