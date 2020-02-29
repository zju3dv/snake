We have been sorting out the code and the pretrained models. Please stay tuned.

# Deep Snake for Real-Time Instance Segmentation

![city](assets/snake_city.png)

> [Deep Snake for Real-Time Instance Segmentation](https://arxiv.org/pdf/2001.01629v2.pdf)  
> Sida Peng, Wen Jiang, Huaijin Pi, Xiuli Li, Hujun Bao, Xiaowei Zhou  
> CVPR 2020

Any questions or discussions are welcomed!

## Installation

### Set up the python environment

```
conda create -n snake python=3.7
conda activate snake

# install torch 1.1 built from cuda 9.0
pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/stable

pip install Cython==0.28.2
pip install -r requirements.txt

# install apex
cd
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```

### Compile cuda extensions under `lib/csrc`

```
ROOT=/path/to/snake
cd $ROOT/lib/csrc
export CUDA_HOME="/usr/local/cuda-9.0"
cd dcn_v2
python setup.py build_ext --inplace
cd ../extreme_utils
python setup.py build_ext --inplace
cd ../roi_align_layer
python setup.py build_ext --inplace
```

### Set up datasets

#### Cityscapes

1. Download the Cityscapes dataset (leftImg8bit\_trainvaltest.zip) from the official [website](https://www.cityscapes-dataset.com/downloads/).
2. Download the processed annotation file [cityscapes_anno.tar.gz](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EcaFL3ZLC5VOvR5HupOgHEMByzgiZ0iLpPW0rAb1i57Ytw?e=tocgyq).
3. Organize the dataset as the following structure:
    ```
    ├── /path/to/cityscapes
    │   ├── annotations
    │   ├── coco_ann
    │   ├── leftImg8bit
    │   ├── gtFine
    ```
3. Generate `coco_img`.
	```
	mkdir -p coco_img/train
	cp leftImg8bit/train/*/* coco_img/train
	cp leftImg8bit/val/*/* coco_img/val
	cp leftImg8bit/test/*/* coco_img/test
	```
4. Create a soft link:
    ```
    ROOT=/path/to/snake
    cd $ROOT/data
    ln -s /path/to/cityscapes cityscapes
    ```

#### Kitti

1. Download the Kitti dataset from the official [website](http://www.cvlibs.net/download.php?file=data_object_image_2.zip).
2. Download the annotation file `instances_train.json` and `instances_val.json` from [Kins](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset).
3. Organize the dataset as the following structure:
	```
    ├── /path/to/kitti
    │   ├── testing
    │   │   ├── image_2
    │   │   ├── instance_val.json
    │   ├── training
    │   │   ├── image_2
    │   │   ├── instance_train.json
    ```
4. Create a soft link:
    ```
    ROOT=/path/to/snake
    cd $ROOT/data
    ln -s /path/to/kitti kitti
    ```

#### Sbd

1. Download the Sbd dataset at [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EV2P-6J0s-hClwW8uZy1ZXYBPU0XwR7Ch7EBGOG2vfACGQ?e=wpyE2M).
2. Create a soft link:
    ```
    ROOT=/path/to/snake
    cd $ROOT/data
    ln -s /path/to/sbd sbd
    ```

## Testing

### Testing on Cityscapes

1. Download the pretrained model [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EX6rAwkK7jBEp7LxKbYIjAkB0QCFjBL4Ov6_aaK1zZFfrA?e=fRWG2x) and put it to `$ROOT/data/model/rcnn_snake/long_rcnn/197.pth`.
2. Test:
    ```
    # use coco evaluator
    python run.py --type evaluate --cfg_file configs/city_rcnn_snake.yaml
    # use the cityscapes offical evaluator
    python run.py --type evaluate --cfg_file configs/city_rcnn_snake.yaml test.dataset CityscapesVal
    ```

### Testing on Kitti

1. Download the pretrained model [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/ERrNrpFPg71HmaegOIqypFkBzqeYn84RF5Sq9dUZM7nsbg?e=bQZ8bp) and put it to `$ROOT/data/model/snake/kins/149.pth`.
2. Test:
    ```
    python run.py --type evaluate --cfg_file configs/kins_snake.yaml test.dataset KinsVal
    ```

### Testing on Sbd

1. Download the pretrained model [here](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EVIoAulD8ORAli3qjdPBMOoBbRTHaxhPHn_a76EznL_W-g?e=EzQQS1) and put it to `$ROOT/data/model/snake/Sbd/149.pth`.
2. Test:
    ```
    python run.py --type evaluate --cfg_file configs/sbd_snake.yaml test.dataset SbdVal
    ```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{peng2020deep,
  title={Deep Snake for Real-Time Instance Segmentation},
  author={Peng, Sida and Jiang, Wen and Pi, Huaijin and Li, Xiuli and Bao, Hujun and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2020}
}
```
