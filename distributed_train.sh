#!/bin/bash
NUM_PROC=$1
shift
#python train.py "$@"
python train.py E:\SUBSimages\classification1 -b 4 --model resnetrs420PS --epochs 200 --lr 0.02
python train.py E:\SUBSimages\classification1 -b 4 --model resnetrs350PS --epochs 200 --lr 0.02
python train.py E:\SUBSimages\classification1 -b 4 --model resnetrs270PS --epochs 200 --lr 0.02
python train.py E:\SUBSimages\classification1 -b 4 --model resnetrs200PS --epochs 200 --lr 0.02
python train.py E:\SUBSimages\classification1 -b 4 --model resnetrs152PS --epochs 200 --lr 0.02
python train.py E:\SUBSimages\classification1 -b 4 --model resnetrs101PS --epochs 200 --lr 0.02
python train.py E:\SUBSimages\classification1 -b 4 --model resnetrs50PS --epochs 200 --lr 0.02
python train.py E:\SUBSimages\classification1 -b 4 --model resnet200PS --epochs 200 --lr 0.02
python train.py E:\SUBSimages\classification1 -b 4 --model resnet152PS --epochs 200 --lr 0.02
python train.py E:\SUBSimages\classification1 -b 4 --model resnet101PS --epochs 200 --lr 0.02
python train.py E:\SUBSimages\classification1 -b 4 --model resnet50PS --epochs 200 --lr 0.02

