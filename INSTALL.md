### Set up the python environment

```
conda create -n snake python=3.7
conda activate snake

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 9.0, install torch 1.1 built from cuda 9.0
pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/stable

pip install Cython==0.28.2
pip install -r requirements.txt

# install apex
cd
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 39e153a3159724432257a8fc118807b359f4d1c8
export CUDA_HOME="/usr/local/cuda-9.0"
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
    │   │   ├── instances_val.json
    │   ├── training
    │   │   ├── image_2
    │   │   ├── instances_train.json
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
