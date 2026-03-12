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
### DDQN Non-Stationary Optimizer Sweep
Run from repo root:
```
python3 ddqn_nonstationary_sweep.py \
  --domains cartpole frozenlake \
  --optimizers sgd sgd_momentum sgd_nag adam adamw rmsprop muon \
  --output-dir "results/ddqn_nonstationary_sweep_$(date +%Y%m%d_%H%M%S)"
```

Default training is step-budgeted:
```
--cartpole-train-steps 300000
--frozenlake-train-steps 1000000
--cartpole-episodes 0
--frozenlake-episodes 0
```

The script writes:
```
results/<output_dir>/run_YYYYMMDD_HHMMSS/
```

Key files in each run folder:
```
raw_results.csv
summary_training.csv
summary_shifted_envs.csv
all_histories.csv
all_evals.csv
all_shift_evals.csv
manifest.json
```

### Common run examples
CartPole-only, 100k steps:
```
python3 ddqn_nonstationary_sweep.py \
  --domains cartpole \
  --optimizers sgd sgd_momentum sgd_nag adam adamw rmsprop muon \
  --seeds $(seq 0 29) \
  --cartpole-episodes 0 \
  --cartpole-train-steps 100000 \
  --output-dir "results/ddqn_cartpole_100k_$(date +%Y%m%d_%H%M%S)"
```

CartPole-only with Muon spectrum + momentum diagnostics:
```
python3 ddqn_nonstationary_sweep.py \
  --domains cartpole \
  --optimizers sgd sgd_momentum sgd_nag adam adamw rmsprop muon \
  --seeds $(seq 0 29) \
  --cartpole-episodes 0 \
  --cartpole-train-steps 100000 \
  --muon-spectrum-every 100 \
  --muon-spectrum-topk 0 \
  --adamw-momentum-every 100 \
  --sgd-momentum-log-every 100 \
  --output-dir "results/ddqn_cartpole_100k_diag_$(date +%Y%m%d_%H%M%S)"
```

Adam/AdamW no first-moment momentum vs RMSProp:
```
python3 ddqn_nonstationary_sweep.py \
  --domains cartpole \
  --optimizers adam adamw rmsprop \
  --seeds $(seq 0 29) \
  --cartpole-episodes 0 \
  --cartpole-train-steps 100000 \
  --adam-beta1 0.0 \
  --adamw-beta1 0.0 \
  --adamw-momentum-every 100 \
  --output-dir "results/ddqn_cartpole_100k_adam_adamw_nomom_vs_rmsprop_$(date +%Y%m%d_%H%M%S)"
```

### Plotting
Auto-plotting is enabled by default at end of sweep (`--auto-plot`):
```
plots/comparison/
plots/muon_spectrum/        # only when Muon spectrum logs exist
plots/adamw_momentum/       # only when Adam/AdamW/Muon momentum logs exist
```

Auto-plot options:
```
--auto-plot / --no-auto-plot
--auto-plot-step-bin 100
```

Manual plotting commands (if needed):
```
MPLCONFIGDIR=/tmp/matplotlib venvPAM/bin/python plot_ddqn_comparison.py <RUN_DIR> \
  --output-dir <RUN_DIR>/plots/comparison \
  --step-bin 100
```

```
MPLCONFIGDIR=/tmp/matplotlib venvPAM/bin/python plot_muon_update_spectra.py <RUN_DIR> \
  --output-dir <RUN_DIR>/plots/muon_spectrum \
  --max-singular-indices 0
```

```
MPLCONFIGDIR=/tmp/matplotlib venvPAM/bin/python plot_adamw_momentum.py <RUN_DIR> \
  --output-dir <RUN_DIR>/plots/adamw_momentum
```

```
MPLCONFIGDIR=/tmp/matplotlib venvPAM/bin/python plot_optimizer_momentum_comparison.py <RUN_DIR> \
  --output-dir <RUN_DIR>/plots/momentum_comparison \
  --metric l2
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
