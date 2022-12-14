source ~/.bashrc

conda activate /srv/cvmlp-lab/flash1/xhuang394/conda/hab_new
ARG1=${1:-full}
echo $ARG1
if [ $ARG1 == "easy" ];
	then
	c=tp_srl.yaml
    f=rearrange_easy_place
else
	c=tp_srl_oracle_plan.yaml
    f=rearrange
fi

srun -u python -u habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/hab/$c --run-type eval WRITER_TYPE tb TEST_EPISODE_COUNT ${4:-10000} DATASET_SAVE_PATH /srv/share/xhuang394/share_demos/$f DATASET_SAVE_START ${2:-50000} TASK_CONFIG.SEED ${3:-42} TASK_CONFIG.SIMULATOR.SEED ${3:-42}
