## Introduction

The project aims to quickly implement a deep learning algorithm.

For a deep learning algorithm, we need four parts: dataset, network, trainer, and evaluator. 
1. dataset: provide the data for training or testing.
2. network: the architecture of our algorithm.
3. trainer: define loss functions for training.
4. evaluator: define metrics for evaluation.

With this project, you only need to implement the four parts. The training and testing pipeline has been implemented.

## Project structure

`task` means the name of this algorithm.

### Dataset

The meta data of dataset like `dataset_id` is registered in `lib/datasets/dataset_catalog.py`.

dataset:
```
lib/datasets/
├── dataset_catalog.py
├── $dataset_id/
    └── $task.py
```

### Network

We could define networks for this task under `lib/networks/$task`.

network:
```
lib/networks/
├── $task/
    └── __init__.py
    └── resnet.py
```

### Trainer

A trainer computes the losses for the algorithm.

Define a trainer for this task under `lib/train/trainers/$task`.

trainer:
```
lib/train/
├── trainers/
    └── $task.py
```

### Evaluator

It is very very necessary for us to define a evaluator. It is a feedback for our training strategies.

Evaluate an algorithm `task` on a dataset that is registered with `dataset_id`.

evaluator:
```
lib/evaluators/
├── $dataset_id/
    └── $task.py
```

### Visualizer

Use a visualizer to watch the output of the network or something else.

Define a visualizer under `lib/visualizers`.

```
lib/visualizers/
└── $task.py
```

## Training and testing

Some variables in this project:
1. `$task`: denote the algorithm.
2. `$dataset_name`: denote the dataset used for training or testing, which is registered in `lib/datasets/dataset_catalog.py`.
3. `$model_name`: the model with a specific configuration.

### Testing

Test a model of our algorithm on a dataset: 
```
python run.py --type evaluate test.dataset $dataset_name resume True model $model_name task $task
```

### Training

Some variables during training:
1. `$test_dataset`: the dataset used for evaluation during training.
2. `$train_dataset`: the dataset used for training.
3. `$network_name`: a specific network architecture for our algorithm `$task`.

Train a model of our algorithm on a dataset for 140 epochs and evaluate it every 5 epochs:

```
python train_net.py test.dataset $test_dataset train.dataset $train_dataset resume False model $model_name task $task network $network_name train.epoch 140 eval_ep 5
```

Some configurations that are frequently revised for training:
1. `train.milestones` and `train.gamma`: multiply the learning rate by `gamma` at some specified epochs.
2. `train.batch_size`: the training batch_size.
3. `train.lr`: the initial learning rate.
4. `train.optim`: the optimizer.
5. `train.warmup`: use the warmup training or not.

More training configurations can be seen in `lib/config/config.py`.

You could define the configuration with the terminal command or yaml files under `configs/`.

During training, we can watch the training losses, testing losses and dataset metrics:

```
cd data/record
tensorboard --logdir $task
```

The losses are defined in the trainer, and the metrics are defined in the evaluator.
