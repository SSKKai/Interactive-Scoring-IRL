## Description

Implementation of **Boosting Feedback Efficiency of Interactive Reinforcement Learning by Adaptive Learning from Scores**.

## Requirements

*   [PyTorch](http://pytorch.org/)
*   [Gym](https://github.com/openai/gym)

Optional Environments:
*   [mujoco-py](https://github.com/openai/mujoco-py)
*   [Metaworld](https://github.com/rlworkgroup/metaworld)
*   [RLBench](https://github.com/stepjam/RLBench)

We use **hydra** to manage our configs:
```
pip install hydra-core --upgrade
```
We use **pyqt5** to create our scoring GUI:
```
pip install pyqt5
pip install pyqt5-tools
```
This repo only support conducting real human experiment in Metaworld environments.

It seems that pyqt5 may conflict with RLBench(Coppeliasim), so if you want to try real human experiment, you may need to delete qt5 python package and remove `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so` in your `~/.bashrc` file for saving the reference videos.

Others requirements are listed in `requirements.txt`

## Usage
We use **hydra** to manage our hypermaraters. We listed the config file in `\config` for different experiments.
To run them, just specify the `CONFIG_NAME`. For example: `--config-name mujoco-HalfCheetah-fb250` or `--config-name metaworld-ButtonPress-fb500`.
```
python main.py [--config-name CONFIG_NAME]
```
If you want to try your own configurations, you can follow these example config and write your own `.yaml` configuration file. 
We have noted the meaning of every hyperparameter in the config file.

We use wandb to log our experiments data, if you want to use it, please change the hyperparameters `wand_log` to `true`.

We use slightly customized environments for the experiments in RLBench simulator. For the usage of RLBench, you can check https://github.com/SSKKai/RLBench-SAC-with-DenseReward

## Real Human Experiment
We design a scoring GUI for real human experiments. You can try it by running:
```
python main.py --config-name metaworld-ButtonPress-realhuman
```
![image](GUI_screenshot.png)

## Results

We use **wandb** to log our experiment data, you can see them in the following link:

*   [Metaworld ButtonPress with Scoring Noise and 0.5 Scoring Precision](https://wandb.ai/sskk/OPRRL-Metaworld-ButtonPress-Scoring-Noise?workspace=user-sskk)
*   [Mujoco HalfCheetah with Scoring Noise and 0.5 Scoring Precision](https://wandb.ai/sskk/OPRRL-Mujoco-HalfCheetah-with-Scoring-Noise?workspace=user-sskk)
*   [Metaworld ButtonPress Comparison Experiments (including real human experiments)](https://wandb.ai/sskk/OPRRL-Metaworld-ButtonPress-Comparison-Experiments?workspace=user-sskk)
*   [Metaworld SweepInto Comparison Experiments](https://wandb.ai/sskk/OPRRL-Metaworld-SweepInto-Comparison-Experiments?workspace=user-sskk)
*   [Mujoco HalfCheetah Comparison Experiments](https://wandb.ai/sskk/OPRRL-Mujoco-HalfCheetah-Comparison-Experiments?workspace=user-sskk)
*   [Mujoco Ant Comparison Experiments](https://wandb.ai/sskk/OPRRL-Mujoco-Ant-Comparison-Experiments?workspace=user-sskk)


## Acknowledgement

This repo is based on the [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic/tree/SAC_V).

We use the repo [similarity_measures](https://github.com/cjekel/similarity_measures) to compute the DTW trajectory distance.


