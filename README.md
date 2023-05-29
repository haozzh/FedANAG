# Stabilizing and Accelerating Federated Learning on Heterogeneous Data


This directory contains source code for evaluating federated learning with FedANAG on various models and tasks. The code was developed for a paper, "Stabilizing and Accelerating Federated Learning on Heterogeneous Data".

## Requirements

Some pip packages are required by this library, and may need to be installed. For more details, see `requirements.txt`. We recommend running `pip install --requirement "requirements.txt"`.

## Task and dataset summary

Note that we put the dataset under the directory .\federated-learning-master\Folder

<!-- mdformat off(This table is sensitive to automatic formatting changes) -->

| Directory        | Model                               | Task Summary              |
|------------------|-------------------------------------|---------------------------|
| CIFAR-10         | CNN (with two convolutional layers) | Image classification      |
| CIFAR-100        | CNN (with two convolutional layers) | Image classification      |
| EMNIST           | Logistic Model                      | Digit recognition         |
| Shakespeare      | RNN with 2 LSTM layers              | Next-character prediction |

<!-- mdformat on -->


## Training
In this code, we compare 10 optimization methods: **FedANAG**, **MimeLite**, **SCAFFOLD**, **FedDyn**, **FedProx**, **FedAvg**,**FedGNAG**, **FedAvgM**, **FedLNAG**, and **FedLNAG\***. Those methods use vanilla SGD on clients. To recreate our experimental results for each optimizer, for example, for 100 clients and 10% participation rate, on the cifar100 data set with Dirichlet (0.3) split, run those commands for different methods:



**FedANAG**:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 100 --frac 0.1 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --lr_decay 0.998 --method fed_nesterov --filepath FedANAG_CIFAR100_seed200.txt
```
>📋  ```--beta_``` is hyperparameters for **FedANAG**.

**MimeLite**:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 100 --frac 0.1 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --lr_decay 0.998 --method mime --filepath MimeLite_CIFAR10_CIFAR100_seed200.txt
```
>📋  ```--beta_``` is hyperparameters for **MimeLite**.

**SCAFFOLD**:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 100 --frac 0.1 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method scaffold --filepath scaffold_CIFAR100_seed200.txt
```

**FedDyn**:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 100 --frac 0.1 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --weigh_delay 0.001 --lr_decay 0.998 --coe 0.1 --method feddyn --filepath feddyn_CIFAR100_CIFAR100_seed200.txt
```
>📋  ```--coe``` is hyperparameters for **FedDyn**.

**FedProx**:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 100 --frac 0.1 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0 --weigh_delay 0.001 --mu 0.001 --lr_decay 0.998 --method fedprox --filepath fedprox_CIFAR100_seed200.txt
```
>📋  ``--mu``` is hyperparameters for **FedProx**.

**FedAvg**:
```
python main_fed.py --seed 100 --gpu 0 --epochs 2000  --num_users 100 --frac 0.1 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --weigh_delay 0.001 --lr_decay 0.998 --method fedavg --filepath fedavg_CIFAR100_seed200.txt
```


**FedGNAG**:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 100 --frac 0.1 --dataset CIFAR100 --local_ep 5 --local_bs 50 --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --lr_decay 0.998 --method fedavg_gnag --filepath FedGNAG_CIFAR100_seed200.txt
```
>📋  ```--beta_``` is hyperparameters for **FedGNAG**.

**FedAvgM**:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 100 --frac 0.1 --dataset CIFAR100 --local_ep 5 --local_bs 50  --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --lr_decay 0.998 --method fedavgm --filepath fedavgm_CIFAR100_seed200.txt
```
>📋  ```--beta_``` is hyperparameters for **FedAvgM**.

**FedLNAG**:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 100 --frac 0.1 --dataset CIFAR100 --local_ep 5 --local_bs 50 --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001  --lr_decay 0.998 --method fed_localnesterov --filepath fed_localnesterov_CIFAR100_seed200.txt
```
>📋  ```--beta_``` is hyperparameters for **FedLNAG**.

**FedLNAG\***:
```
python main_fed.py --seed 200 --gpu 0 --epochs 2000  --num_users 100 --frac 0.1 --dataset CIFAR100 --local_ep 5 --local_bs 50 --bs 50 --rule_arg 0.3 --lr 0.1 --beta_ 0.9 --weigh_delay 0.001 --lr_decay 0.998 --method fed_localnesterov_nocom --filepath fed_localnesterov_nocom_CIFAR100_seed200.txt
```
>📋  ```--alpha``` is hyperparameters for **FedLNAG\***.


## Other hyperparameters and reproducibility

All other hyperparameters are set by default to the values used in the `Experiment Details` of our Appendix. This includes the batch size, the number of clients per round, the number of client local updates, local learning rate, and model parameter flags. While they can be set for different behavior (such as varying the number of client local updates), they should not be changed if one wishes to reproduce the results from our paper. 



