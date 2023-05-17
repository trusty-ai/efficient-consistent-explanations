# Code for "Trade-off Between Efficiency and Consistency for Removal-based Explanations"


## Environment requirement
We run our code on a Ubuntu 18.04 machine with 4 Nvidia 2080 Ti graphic cards.
The CUDA version should be `10.2`. 
You can use `nvcc --version` to check CUDA version.

We recommend to use conda environment and the environment file is provided. 
You can run the following code to build the environment.

```
conda env create -f environment.yml
conda activate consistent
python -m spacy download en_core_web_sm
```

## SST2 experiments
### Train a model
```
cd SST2
python sst2_train.py --long_sentence_trucate 50
```
It will save model files which are later to be interpreted.

### Second order Harmonica
```
python sst2-load-second.py
```

### Third order Harmonica
```
python sst2-load-third.py
```

### Low-degree algorithm
```
# for second order
python sst2-load-low-degree-second.py --samples_min 2000
# for third order
python sst2-load-low-degree-third.py --samples_min 2000
```
You can use `samples_min` to specify the sample size. 
In our paper, we use 2000, 4000, 6000, 8000 and 10000.

### Harmonica-local
```
# for second order
python sst2-load-second-local.py
# for third order
python sst2-load-third-local.py
```

### Harmonica-anchor
```
python sst2-load-second-partial-preciselasso.py --samples_min 6000 --n 3 --ep_consistent_loss 0 
```

### Harmonica-anchor-constrained
```
python sst2-load-second-partial-preciselasso.py --samples_min 6000 --n 3 --ep_consistent_loss 0.1 
```

### Baseline algorithms
```
# for LIME
python sst2_baselines_first_order.py --algorithm lime
# for Integrated gradients
python sst2_baselines_first_order.py --algorithm ig
# for KernelSHAP
python sst2_baselines_first_order.py --algorithm ks
# for Integrated Hessians
python sst2_baselines_second_order.py --algorithm ih
# for Shapley Taylor Index (second order)
python sst2_baselines_second_order.py --algorithm shaptaylor
# for Faith-Shap (second order)
python sst2_baselines_second_order.py --algorithm faithshap
```

## IMDb experiments
### Train a model
```
cd IMDB
python torchtext_train.py
```
It will save a model file which is latter to be interpreted.

### Second order Harmonica
```
python imdb_classification-load-second.py
```
### Third order Harmonica
```
python imdb_classification-load-third.py
```

### Harmonica-anchor

```
python imdb_classification-partial-preciselasso.py --ep_consistent_loss 0 --n 3 --samples_min 6000```
```


### Baseline algorithms
```
# for LIME
python imdb_baselines_first_order.py --algorithm lime
# for Integrated gradients
python imdb_baselines_first_order.py --algorithm ig
# for KernelSHAP
python imdb_baselines_first_order.py --algorithm ks
# for Integrated Hessians
python imdb_baselines_second_order.py --algorithm ih
# for Shapley Taylor Index (second order)
python imdb_baselines_second_order.py --algorithm shaptaylor
# for Faith-Shap (second order)
python imdb_baselines_second_order.py --algorithm faithshap
```

## ImageNet experiments
```
cd Image
```
We use the official PyTorch model so we do not need to train a classifier.
### Second order Harmonica
```
python image-load-second.py --n_superpixels 16 --samples_min 128
```


### Third order Harmonica
```
python image-load-third.py --n_superpixels 16 --samples_min 128
```

### Harmonica-anchor
```
python image-load-second-partial-preciselasso.py --model resnet101 --n_superpixels 16 --samples_min 4000 --n 3 --ep_consistent_loss 0
```

### Baseline algorithms
```
# for LIME
python image_baselines_first_order.py --algorithm lime --n_superpixels 16 --samples_min 2000
# for Integrated gradients
python image_baselines_first_order.py --algorithm ig --n_superpixels 16 --samples_min 2000
# for KernelSHAP
python image_baselines_first_order.py --algorithm ks --n_superpixels 16 --samples_min 2000
# for Integrated Hessians
python image_baselines_second_order.py --algorithm ih --n_superpixels 16 --samples_min 2000
# for Shapley Taylor Index (second order)
python image_baselines_second_order.py --algorithm shaptaylor --n_superpixels 16 --samples_min 2000
# for Faith-Shap (second order)
python image_baselines_second_order.py --algorithm faithshap  --n_superpixels 16 --samples_min 2000 --model resnet101
```

## Draw figures in the paper
We use jupyters notebooks to draw all the figures in our paper. 
The above python files save the interpretation error and truthful gap information.
After running the python files, you should specify the path of the saved files in jupyter notebooks accroding to your running directory.
And then run the jupyter notebooks and you will get all the figures.



