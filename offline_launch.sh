export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
source ~/.bashrc
conda activate /srv/cvmlp-lab/flash1/xhuang394/conda/hab_new
export TMPDIR=~/tmp/
if [ $1 == "easy" ];
	then
	CONFIG="transformer_easy_offline.yaml"
else
	CONFIG="transformer_offline.yaml"
fi

if [ $2 == "eval" ];
	then
	MODE="eval"
	wandb offline
else
	MODE="train"
	wandb online
fi
srun -u python -u habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/hab/$CONFIG --run-type $MODE
