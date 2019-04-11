# COMP4211 Assignment 2

Ng Chi Him
SID 20420921
chngax@connect.ust.hk

## Usage

The three tasks have been implemented on separate Jupyter Notebooks in Anaconda environment with Python 3.6.

To setup and activate the environment, run:

```sh
conda create -n comp4211 python=3.6 anaconda
conda install -n comp4211 -c pytorch pytorch torchvision
conda install -n comp4211 tensorflow tensorboardX
conda active comp4211
```

Then, launch Jupyter Notebook and open files using web ui:

```sh
jupyter notebook
```

Logs generated in the submission is stored in `/logs` and can be viewed by tensorboard:

```sh
tensorboard --logdir ./logs/
```

## Provided code and Dataset

The three provided files (`pa2_sample_code.py`, `pretrained_encoder.pt` and `train_test_split.npz`) are stored in root directory of the project. They are included in the code submission but feel free to replace them with official copy when grading.

As instructed, the dataset was not included in code submission. To obtain the dataset, run `pa2_sample_code.py`. (or place local copy to `/data` when grading, folder structure is not modified.)
