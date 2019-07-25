# BIC
pytorch implementation of "Large Scale Incremental Learning" from https://arxiv.org/abs/1905.13260

# Dataset
Download Cifar100 dataset from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

Put meta, train, test into ./cifar100

# Train
python main.py


Result

|    |  20  |  40  |  60  |  80  |  100  |
| ---- | ---- | ---- | ---- | ---- | ---- |
|  Paper  | 85.20 | 74.59 | 66.76 | 60.14 | 55.55 |
|  Implementation  | 83.00 | 69.90 | 63.08 | 57.11 | 53.7 |
