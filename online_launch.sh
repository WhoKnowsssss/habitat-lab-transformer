export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
source ~/.bashrc
conda activate /srv/cvmlp-lab/flash1/xhuang394/conda/hab_new
export TMPDIR=~/tmp/
if [ $1 == "train" ];
	then
	wandb offline
else
	wandb offline
fi
srun -u python -u habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/hab/transformer_online.yaml --run-type $1
