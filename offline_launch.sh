export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
source ~/.bashrc

conda activate /srv/cvmlp-lab/flash1/xhuang394/conda/hab_new
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TMPDIR=~/tmp/
if [ $1 == "all" ];
	then
	CONFIG="transformer_all_offline.yaml"
else
	CONFIG="transformer_offline.yaml"
fi

if [ $2 == "eval" ];
	then
	MODE="eval"
	wandb online
else
	MODE="train"
	wandb online
fi

srun -u python -u habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/hab/$CONFIG --run-type $MODE TASK_CONFIG.SEED ${3:-42} TASK_CONFIG.SIMULATOR.SEED ${3:-42}
