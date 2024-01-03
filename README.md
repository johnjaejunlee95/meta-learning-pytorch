# Meta-Learning-pytorch



## Setting

``````
conda create env -y -n metalearning python=3.9
conda activate metalearning
bash install.sh
``````



## Datasets

#### miniImageNet Source:

https://drive.google.com/file/d/1-8XtrPWViumNpgT4u53TYu3qWqJ0r-Aa/view

#### tieredImageNet Image Source:

https://drive.google.com/file/d/16H2Hlv3HE0P3cVHGr_es36RnWV_zrCjH/view




## Training

### MAML

##### Run Omniglot:

```
python train_maml_omniglot.py -- model [MODEL] --epoch 15000 --num_ways [5 or 20] --num_shots [1 or 5] --batch_size [32 or 16] --update_lr [0.4 or 0.1] --update_step [1 or 5] --update_step_test [3 or 5]
```

##### Run mini-ImageNet or tiered-ImageNet

```
python train.py --model [MODELS] --datasets [DATASETS] --epoch 60000 --num_shots [1 or 5] --batch_size [4 or 2]
```

### Prototypical Networks

##### Run mini-ImageNet or tiered-ImageNet

```
python train.py --epoch 20000 --batch_size 1 --model [MODEL] --datasets [DATASETS] 
```

### Reptile

##### Run mini-ImageNet or tiered-ImageNet (based on the Paper experimental settings)

```
python train.py --model [MODELS] --datasets [DATASETS] --epoch 100000 --num_ways 5 --training_shots 15 --num_shots 5 --batch_size 5 --inner_batch_size 10 --eval_inner_batch_size 15 --max_test_task 50 --num_sampling 100
```
