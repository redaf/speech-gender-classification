#!/bin/bash
[ ! -d "models" ] && python train.py
python predict.py $1
