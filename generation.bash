source ~/.bashrc

conda activate /srv/cvmlp-lab/flash1/xhuang394/conda/habitat_new
if [ $1 == "easy" ];
	then
	c=tp_srl.yaml
    f=rearrange_easy
else
	c=tp_srl_oracle_plan.yaml
    f=rearrange
fi
srun -u python -u habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/hab/$c --run-type eval WRITER_TYPE tb TEST_EPISODE_COUNT 5000 DATASET_SAVE_PATH /srv/share/xhuang394/share_demos/$f TASK_CONFIG.SEED 42 TASK_CONFIG.SIMULATOR.SEED 42
