import os, pickle, time
from this import d
from typing import Any, ClassVar, Dict, List, Tuple, Union, Optional
import itertools as its
from collections import deque

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
import numba


from habitat import Config, logger
from habitat_baselines.common.tensor_dict import TensorDict


def read_dataset(
    config: Config,
    verbose: bool,
    rng: np.random.Generator,
    context_length: int = 30,
):
    obss = []
    actions = []
    done_idxs = []
    stepwise_returns = []

    skill_dict = {
        "nav": 0,
        "pick": 1,
        "place": 2,
        "open_cab": 3,
        "open_fridge": 3,
        "reset_arm": -1,
        "wait": -1,
    }

    path = config.trajectory_dir
    dataset_size = config.dataset_size
    if dataset_size != -1:
        filenames = os.listdir(path)[:dataset_size]
    else:
        filenames = os.listdir(path)
    # filenames = ['{}.pt'.format(i*10) for i in range(1,51)]

    if verbose:
        logger.info("Trajectory Files: {}".format(filenames))

    transitions_per_buffer = np.zeros(len(filenames), dtype=int)
    num_trajectories = 0
    previous_done = 0
    while len(obss) < config.files_per_load:
        buffer_num = rng.choice(np.arange(len(filenames)), 1, replace=False)[0]
        i = transitions_per_buffer[buffer_num]
        if verbose:
            logger.info("Loading from buffer {}".format(buffer_num, i))
        file = os.path.join(path, filenames[buffer_num])
        try:
            file = os.readlink(file)
            # print('symbol link', file)
        except Exception as e:
            # print('no link', file)
            pass
        if os.path.exists(file):
            import time

            s = time.perf_counter()
            dataset_raw = torch.load(file, map_location=torch.device("cpu"))
            # print("dataset load time", time.perf_counter() - s)

            temp_obs = np.array(dataset_raw["obs"])
            temp_actions = torch.stack(dataset_raw["actions"]).numpy()[:, :12]
            temp_stepwise_returns = torch.cat(dataset_raw["rewards"]).numpy()
            temp_dones = torch.cat(dataset_raw["masks"]).numpy()
            temp_infos = np.array(dataset_raw["infos"])

            # if len(temp_actions) == len(temp_obs):
            #     temp_actions[:-1,[8,9]] = np.stack([temp_obs[i]['oracle_nav_executed_action'] for i in range(len(temp_obs))])[1:]
            # else:
            #     temp_actions[:,[8,9]] = np.stack([temp_obs[i]['oracle_nav_executed_action'] for i in range(len(temp_obs))])[1:]
            #     temp_obs = temp_obs[:-1]

            # ==================== Only Arm Action Phase ===================
            # stepwise_idx = np.argwhere(np.all(temp_actions[:,8:10] == 0, axis=-1) & temp_dones == True).squeeze()

            # temp_obs = np.delete(temp_obs, stepwise_idx, 0)
            # temp_actions = np.delete(temp_actions, stepwise_idx, 0)
            # temp_stepwise_returns = np.delete(temp_stepwise_returns, stepwise_idx, 0)
            # temp_dones = np.delete(temp_dones, stepwise_idx, 0)

            # ===================== Only Nav Pick ====================
            # if True: #int(filenames[buffer_num][:-3]) <= 50000
            #     temp_done_idxs = np.argwhere(temp_dones == False).reshape(-1) + 1
            #     temp_start_idxs = np.roll(temp_done_idxs, 1)
            #     temp_start_idxs[0] = 0
            #     temp_nav_phase = np.all(temp_actions[:,:7] == 0, axis=-1).astype(np.int8)
            #     temp_nav_phase = np.nonzero((temp_nav_phase[1:] - temp_nav_phase[:-1]) > 0)[0]
            #     temp_nav_phase = np.concatenate([temp_nav_phase, temp_done_idxs[-1:]-1])
            #     temp_nav_place_idx = np.searchsorted(temp_nav_phase, temp_start_idxs, side='left')
            #     temp_nav_phase = temp_nav_phase[temp_nav_place_idx].reshape(-1)
            #     stepwise_idx = np.concatenate([np.arange(temp_nav_phase[i] + 1 , temp_done_idxs[i]) for i in range(temp_nav_phase.shape[0])])

            #     temp_dones[temp_nav_phase] = False
            #     temp_obs = np.delete(temp_obs, stepwise_idx, 0)
            #     temp_actions = np.delete(temp_actions, stepwise_idx, 0)
            #     temp_stepwise_returns = np.delete(temp_stepwise_returns, stepwise_idx, 0)
            #     temp_dones = np.delete(temp_dones, stepwise_idx, 0)

            # ===================== Only Nav Place ====================
            # temp_done_idxs = np.argwhere(temp_dones == False).reshape(-1) + 1
            # temp_start_idxs = np.roll(temp_done_idxs, 1)
            # temp_start_idxs[0] = 0
            # temp_nav_phase = np.all(temp_actions[:,:7] == 0, axis=-1).astype(np.int8)
            # temp_nav_phase = np.nonzero((temp_nav_phase[1:] - temp_nav_phase[:-1]) > 0)[0]
            # temp_nav_phase = np.concatenate([temp_nav_phase, temp_done_idxs[-1:]-1])
            # temp_nav_place_idx = np.searchsorted(temp_nav_phase, temp_start_idxs, side='left')
            # temp_nav_phase = temp_nav_phase[temp_nav_place_idx].reshape(-1)
            # stepwise_idx = np.concatenate([np.arange(temp_start_idxs[i], temp_nav_phase[i]) for i in range(temp_nav_phase.shape[0])])

            # temp_dones[temp_nav_phase] = False
            # temp_obs = np.delete(temp_obs, stepwise_idx, 0)
            # temp_actions = np.delete(temp_actions, stepwise_idx, 0)
            # temp_stepwise_returns = np.delete(temp_stepwise_returns, stepwise_idx, 0)
            # temp_dones = np.delete(temp_dones, stepwise_idx, 0)

            # ==================== Only Successful Episodes ===================
            # stepwise_idx = np.argwhere(temp_infos == False).squeeze()

            # temp_obs = np.delete(temp_obs, stepwise_idx, 0)
            # temp_actions = np.delete(temp_actions, stepwise_idx, 0)
            # temp_stepwise_returns = np.delete(temp_stepwise_returns, stepwise_idx, 0)
            # temp_dones = np.delete(temp_dones, stepwise_idx, 0)

            temp_done_idxs = np.argwhere(temp_dones == False).reshape(-1) + 1
            if len(temp_done_idxs) == 0:
                print(
                    "\n\nEVEN NO SUCCESS IN THIS FILE!!!!!!",
                    temp_start_idxs,
                    temp_nav_phase,
                )
                continue

            # ==================== Only Episodes that are larger than context_length ===================
            temp_start_idxs = np.roll(temp_done_idxs, 1)
            temp_start_idxs[0] = 0

            idx = np.nonzero(
                temp_done_idxs[:] - temp_start_idxs[:] < context_length
            )[0]
            if len(idx) > 0:

                stepwise_idx = np.concatenate(
                    [
                        np.arange(temp_start_idxs[i], temp_done_idxs[i])
                        for i in idx
                    ]
                )

                temp_obs = np.delete(temp_obs, stepwise_idx, 0)
                temp_actions = np.delete(temp_actions, stepwise_idx, 0)
                temp_stepwise_returns = np.delete(
                    temp_stepwise_returns, stepwise_idx, 0
                )
                temp_dones = np.delete(temp_dones, stepwise_idx, 0)

            temp_done_idxs = np.argwhere(temp_dones == False).reshape(-1) + 1

            # #==================== Categorize Gripper Action ===================
            temp_actions = np.clip(temp_actions, -1, 1)
            # temp_pick_action = temp_actions[:,7]
            # temp_pick_action = np.nonzero((temp_pick_action[1:] - temp_pick_action[:-1]) < -0.1)[0] + 1
            # temp_place_idx = np.searchsorted(temp_pick_action, temp_done_idxs, side='right') - 1
            # temp_pick_action = temp_pick_action[temp_place_idx].reshape(-1)
            # stepwise_idx = np.concatenate([np.arange(temp_pick_action[i] , temp_done_idxs[i]) for i in range(temp_pick_action.shape[0])])
            # temp_actions[stepwise_idx, 7] = 0

            temp_actions[:, 10] = 0
            temp_pick_action = np.stack(
                [temp_obs[i]["is_holding"] for i in range(len(temp_obs))]
            )
            change = temp_pick_action[1:-1] - temp_pick_action[:-2]
            # if int(filenames[buffer_num][:-3]) < 50000:
            #     ii = np.where(change > 0)[0]
            #     ii = [np.arange(iii-20, min(iii+30, len(temp_actions)-1)) for iii in ii]
            #     ii = np.concatenate(ii)
            #     temp_actions[ii,7] = 2
            # elif int(filenames[buffer_num][:-3]) < 100000:
            #     ii = np.where(change < 0)[0]
            #     ii = [np.arange(iii, min(iii+20, len(temp_actions)-1)) for iii in ii]
            #     ii = np.concatenate(ii)
            #     temp_actions[ii,7] = 1
            # else:
            if True:
                ii = np.where(change < 0)[0]
                try:
                    ii = np.concatenate(
                        [
                            np.arange(
                                iii, min(iii + 20, len(temp_actions) - 1)
                            )
                            for iii in ii
                        ]
                    )
                    temp_actions[ii, 10] = 1
                except:
                    pass
                ii = np.where(change > 0)[0]
                try:
                    ii = np.concatenate(
                        [
                            np.arange(
                                iii - 20, min(iii + 30, len(temp_actions) - 1)
                            )
                            for iii in ii
                        ]
                    )
                    temp_actions[ii, 10] = 2
                except:
                    pass

            # ==================== Add Noise ========================
            # for i in range(len(temp_obs)):
            #     for k in temp_obs[i].keys():
            #         temp_obs[i][k] += np.random.randn(*temp_obs[i][k].shape).astype(np.float32) * 0.05

            # ==================== Simple Filtering =================
            # mask = np.all(temp_actions[:,:7] == 0, axis=-1)
            # temp_actions[~mask,8:10] = 0

            # ==================== Planner Targets ==================
            try:
                for i in range(len(temp_obs)):
                    temp_obs[i]["skill"] = skill_dict[temp_infos[i]["skill"]]
                    if temp_obs[i]["skill"] == -1:
                        temp_obs[i]["skill"] = temp_obs[i - 1]["skill"]
            except:
                continue

            for i in range(len(temp_obs)):
                if temp_obs[i]["skill"] == 0 and temp_obs[i]["is_holding"] == 1:
                    # temp_actions[i,8] = 0
                    temp_obs[i]["skill"] = 4
            
                if temp_obs[i]["skill"] == 4 or temp_obs[i]["skill"] == 0:
                    temp_actions[i,:7] = 0


            # for i in range(len(temp_obs)):
            #     if temp_obs[i]['skill'] != 3:
            #         temp_obs[i]['skill'] = int(temp_obs[i]['is_holding'])

            obss += [temp_obs]
            actions += [temp_actions]
            done_idxs += [temp_done_idxs + previous_done]
            previous_done += len(temp_actions)
            stepwise_returns += [temp_stepwise_returns]

    actions = np.concatenate(actions)
    obss = np.concatenate(obss).tolist()
    stepwise_returns = np.concatenate(stepwise_returns)
    done_idxs = np.concatenate(done_idxs)

    rtg, timesteps = _timesteps_rtg(done_idxs, stepwise_returns)

    if verbose:
        logger.info(
            "In this load, max rtg is {}, max timestep is {}. ".format(
                rtg.max().round(2), timesteps.max()
            )
        )

    obss = TensorDict.from_tree(default_collate(obss))
    actions = torch.from_numpy(actions).to(torch.float32)
    rtg = torch.from_numpy(rtg).to(torch.float32)
    timesteps = torch.from_numpy(timesteps).to(torch.int64)
    return obss, actions, done_idxs, rtg, timesteps


@numba.jit(nopython=True, parallel=True)
def _timesteps_rtg(done_idxs, stepwise_returns):
    rtg = np.zeros_like(stepwise_returns)
    timesteps = np.zeros(len(stepwise_returns), dtype=np.int64)
    start_index = np.concatenate(
        (np.array([0], dtype=np.int64), done_idxs[:-1])
    )
    for i in numba.prange(len(done_idxs)):
        start = start_index[i]
        done = done_idxs[i]
        curr_traj_returns = stepwise_returns[start:done]

        for j in numba.prange(done - start):
            rtg[j + start] = np.sum(curr_traj_returns[j:])

        timesteps[start:done] = np.arange(done - start)
    return rtg, timesteps


def producer(
    config: Config,
    rng: np.random.Generator,
    deque: deque,
    verbose: bool,
    context_length: int = 30,
):
    import time

    while True:
        if len(deque) < config.queue_size:
            s = time.perf_counter()
            deque.append(read_dataset(config, verbose, rng, context_length))
            # print("dataset loaded, ", time.perf_counter() - s)
            time.sleep(1)
        else:
            time.sleep(1)
