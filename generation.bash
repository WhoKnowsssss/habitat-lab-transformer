source ~/.bashrc

conda activate /srv/cvmlp-lab/flash1/xhuang394/conda/habitat_new

srun -u python -u habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/hab/tp_srl.yaml --run-type eval BASE_TASK_CONFIG_PATH configs/tasks/rearrange/rearrange_easy.yaml WRITER_TYPE tb TEST_EPISODE_COUNT 5000 DATASET_SAVE_PATH /srv/share/xhuang394/share_demos/rearrange_easy_5k TASK_CONFIG.SEED 42 TASK_CONFIG.SIMULATOR.SEED 42
