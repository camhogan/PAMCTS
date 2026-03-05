# PAMCTS
## Decision Making in Non-Stationary Environments with Policy-Augmented Search

## Installation
```
pip install -r requirements.txt
```

Installation Error Fix: ModuleNotFoundError: No module named 'distutils.cmd'
```
sudo apt-get install python3.9-distutils
sudo apt-get install python3-apt
```

Box2d Installation Error Fix: error: command 'swig' failed: No such file or directory
```
sudo apt-get update
sudo apt-get -y install swig
```

Box2d Installation Error Fix: error: command '/usr/bin/x86_64-linux-gnu-gcc' failed with exit code 1
```
sudo apt-get update
sudo apt-get install python3.9-dev
sudo apt-get install g++
sudo apt-get install gcc
```

Box2d AttributeError: module '_Box2D' has no attribute 'RAND_LIMIT_swigconstant'
```
pip install box2d box2d-kengz
```

ERROR: google-auth 2.22.0 has requirement urllib3<2.0, but you'll have urllib3 2.0.4 which is incompatible.
```
pip install --upgrade botocore google-auth
```

## Run Experiments
### DDQN Non-Stationary Optimizer Sweep (CartPole + FrozenLake)
Run from the repo root:
```
./venvPAM/bin/python ddqn_nonstationary_sweep.py \
  --domains cartpole frozenlake \
  --optimizers sgd sgd_momentum sgd_nag adamw rmsprop muon \
  --output-dir results/ddqn_nonstationary_sweep
```


The script writes a timestamped run folder:
```
results/ddqn_nonstationary_sweep/run_YYYYMMDD_HHMMSS/
```

Key output files:
```
summary_training.csv
summary_shifted_envs.csv
all_histories.csv
all_evals.csv
all_shift_evals.csv
```

Plot a single run:
```
env MPLCONFIGDIR=/tmp/mpl ./venvPAM/bin/python plot_ddqn_results.py \
  results/ddqn_nonstationary_sweep/run_YYYYMMDD_HHMMSS
```

Compare two runs in one set of plots (example: baseline optimizer run + MUON run):
```
env MPLCONFIGDIR=/tmp/mpl ./venvPAM/bin/python plot_ddqn_comparison.py \
  results/ddqn_nonstationary_sweep/run_20260302_205258 \
  results/ddqn_nonstationary_sweep_muon/run_YYYYMMDD_HHMMSS \
  --output-dir results/ddqn_nonstationary_comparison_plots
```

### Cartpole:
Train DDQN Agent:
```
python Cartpole/train_masscart_dqn.py
```

Train Alphazero Network:
```
python Cartpole/Alphazero/training.py
```

Run PAMCTS:
```
python Cartpole/cartpole_pamcts.py
```

### FrozenLake:
Train DDQN Agent:
```
python Frozenlake/Network_Weights/DQN_3x3/frozenlake_reorder_dqn_retrain.py
```

Train Alphazero Network:
```
bash Frozenlake/Alphazero_Training/start.sh
```

Run PAMCTS:
```
python Frozenlake/flfl_pamcts.py
```

Run Alphazero:
```
python Frozenlake/flfl_alphazero.py
```

Run PAMCTS Alpha Selection:
```
python flfl_alpha_selection_part1.py
```

### Lunar Lander:
Train DDQN Agent:
```
python LunarLander/network_weights/DDQN/ddqn_agent_train.py
```

Train Alphazero Network:
```
python LunarLander/network_weights/Alphazero_Networks/training.py
```

Run PAMCTS:
```
python LunarLander/pamcts.py
```

Run Alphazero:
```
python LunarLander/alphazero.py
```

Run PAMCTS Alpha Selection:
```
python LunarLander/alpha_selection.py
```

### CliffWalking
Train DDQN Agent:
```
python CliffWalking/ddqn_agent.py
```

Train Alphazero Network:
```
python CliffWalking/alphazero_training.py
```

Run PAMCTS:
```
python CliffWalking/pamcts.py
```

Run Alphazero:
```
python CliffWalking/alphazero.py
```

Run PAMCTS Alpha Selection:
```
python CliffWalking/alpha_selection.py
```
