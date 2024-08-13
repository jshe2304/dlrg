#!/bin/bash

python train_many.py n_cd_samples
python train_many.py k_fine
python train_many.py k_coarse
python train_many.py n_models
