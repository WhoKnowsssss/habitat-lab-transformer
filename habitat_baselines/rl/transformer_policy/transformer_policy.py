#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import time
from typing import Dict, List, Optional, Tuple

import torch
from gym import spaces
from torch import device, nn as nn
import torch.nn.functional as F
import numpy as np

from habitat.config import Config
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.rl.transformer_policy.transformer_model_orig import (
    GPTConfig,
    GPT,
    PlannerGPT,
)

# from habitat_baselines.rl.models.rnn_state_encoder import (
#     build_rnn_state_encoder,
# )
from habitat_baselines.rl.transformer_policy.pure_bc_model import LSTMBC
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import get_num_actions
from habitat_baselines.rl.transformer_policy.action_distribution import (
    ActionDistribution,
)
from habitat_baselines.rl.ppo import NetPolicy
from habitat_baselines.common.tensor_dict import TensorDict

from habitat.core.spaces import ActionSpace, EmptySpace

from .focal_loss import FocalLoss


@baseline_registry.register_policy
class TransformerResNetPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        context_length: int = 30,
        max_episode_step: int = 200,
        n_layer: int = 6,
        n_head: int = 8,
        reg_flags=None,
        model_type: str = "reward_conditioned",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        force_blind_policy: bool = False,
        policy_config: Config = None,
        fuse_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        include_visual_keys = policy_config.include_visual_keys
        self.offline_training = policy_config.offline
        self.train_planner = policy_config.train_planner
        self.train_control = policy_config.train_control
        super().__init__(
            TransformerResnetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                context_length=context_length,
                max_episode_step=max_episode_step,
                model_type=model_type,
                n_head=n_head,
                n_layer=n_layer,
                reg_flags=reg_flags,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                force_blind_policy=force_blind_policy,
                discrete_actions=False,
                fuse_keys=[
                    k for k in fuse_keys if k not in include_visual_keys
                ],
                include_visual_keys=include_visual_keys,
                use_rgb=policy_config.use_rgb,
            ),
            action_space=action_space,
            policy_config=policy_config,
        )
        self.boundaries_mean = torch.linspace(-1, 1, 21).cuda()
        self.boundaries = torch.linspace(-1.025, 1.025, 22).cuda()
        if self.offline_training:
            self.loss_vars = nn.parameter.Parameter(torch.zeros((3,)))
            self.focal_loss = FocalLoss(gamma=5).cuda()
            self.focal_loss_loc = FocalLoss(gamma=5).cuda()
            self.focal_loss_planner = FocalLoss(gamma=5).cuda()
            self.focal_loss_planner_2 = FocalLoss(gamma=5).cuda()
            self.focal_loss_pick = FocalLoss(
                alpha=(1 - torch.tensor([0.8, 0.1, 0.1])), gamma=5
            ).cuda()

        self.action_config = policy_config.ACTION_DIST

        if self.action_distribution_type == "categorical":
            self.len_logit = [21 * 7, 3, 21 * 2]
        elif self.action_distribution_type == "gaussian":
            self.len_logit = [7, 1, 2]
        elif self.action_distribution_type == "mixed":
            self.len_logit = [
                21 * 7 if self.action_config.discrete_arm else 7,
                3,
                21 * 2 if self.action_config.discrete_base else 2,
                # 21  * 21 if self.action_config.discrete_base else 2,
            ]
        else:
            raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space: spaces.Dict,
        action_space,
        orig_action_space=None,
        **kwargs,
    ):
        orig_action_space = ActionSpace(
            {
                "ARM_ACTION": spaces.Dict(
                    {
                        "arm_action": spaces.Box(
                            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
                        ),
                        "grip_action": spaces.Box(
                            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                        ),
                    }
                ),
                "BASE_VELOCITY": spaces.Dict(
                    {
                        "base_vel": spaces.Box(
                            low=-20.0, high=20.0, shape=(2,), dtype=np.float32
                        )
                    }
                ),
                "REARRANGE_STOP": EmptySpace(),
            }
        )
        return cls(
            observation_space=observation_space,
            action_space=orig_action_space,
            hidden_size=config.RL.TRANSFORMER.hidden_size,
            context_length=config.RL.TRANSFORMER.context_length,
            max_episode_step=config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS,
            model_type=config.RL.TRANSFORMER.model_type,
            n_head=config.RL.TRANSFORMER.n_head,
            n_layer=config.RL.TRANSFORMER.n_layer,
            reg_flags=config.RL.TRANSFORMER.reg_flags,
            backbone=config.RL.TRANSFORMER.backbone,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            policy_config=config.RL.POLICY,
            fuse_keys=config.TASK_CONFIG.GYM.OBS_KEYS,
            # fuse_keys=config.RL.GYM_OBS_KEYS
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        (value, action, action_log_probs, rnn_hidden_states,) = super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=deterministic,
        )
        action = action.float()
        if self.action_distribution_type == "mixed":
            if self.action_config.discrete_base:
                action = torch.cat(
                    [action[:, :8], action[:, 8:9] // 21, action[:, 8:9] % 21],
                    dim=-1,
                )
                action[:, 8:10] = self.boundaries_mean[
                    action[:, 8:10].to(torch.long)
                ]
            if self.action_config.discrete_arm:
                action[:, :7] = self.boundaries_mean[
                    action[:, :7].to(torch.long)
                ]
            action[:, 7] = (
                (action[:, 7] == 1).int()
                + 2 * (action[:, 7] == 0).int()
                + 3 * (action[:, 7] == 2).int()
                - 2
            )
            mask = action[:, 7:8] == -1
            action = torch.cat(
                [action, torch.zeros_like(mask.float())], dim=-1
            )
        # #============= advance hidden state ===============
        mask = ~torch.any((rnn_hidden_states.sum(-1) == 0), -1)
        rnn_hidden_states[mask] = rnn_hidden_states[mask].roll(-1, 1)
        rnn_hidden_states[mask, -1, :] = 0

        # #============= reset arm ===============
        B = rnn_hidden_states.shape[0]
        if not hasattr(self, "reset_mask"):
            self.reset_mask = torch.zeros(
                B, dtype=torch.bool, device=rnn_hidden_states.device
            )

        if not hasattr(self, "holding_mask"):
            self.holding_mask = torch.zeros(
                B, dtype=torch.bool, device=rnn_hidden_states.device
            )

        if not hasattr(self, "_initial_delta"):
            self._initial_delta = torch.zeros(
                (B, 7), dtype=torch.float, device=rnn_hidden_states.device
            )

        holding_mask = (
            self.holding_mask != observations["is_holding"].reshape(B)
        ) & (self.net.cur_skill_all != 5)
        self.holding_mask = observations["is_holding"].reshape(B)

        self.reset_mask = self.reset_mask | holding_mask | self.net.reset_mask
        self._reset_arm(
            observations,
            action,
            rnn_hidden_states,
            holding_mask | self.net.reset_mask,
        )
        self._back_up(
            observations, action, rnn_hidden_states, holding_mask | self.net.reset_mask
        )

        action[:, 7] = (
            -2
            * (torch.norm(observations["obj_goal_sensor"], dim=-1) < 0.29)
            * observations["is_holding"].reshape(B)
            + 1
            * ((torch.norm(observations["obj_start_sensor"], dim=-1) < 0.59)
            + observations["is_holding"].reshape(B)).bool()
            - 0.01
        )
        mask = action[:, 7] <= -1
        self.gripper_action = action[:, 7]
        action[:, -1] = mask.float()

        # angle = observations["obj_start_gps_compass"][:,1]
        # self.angle = angle
        # mask = (self.net.cur_skill == 5) & (self.net.timeout < 250) & (self.net.timeout > 0)
        # action[mask, 8] = 0.15 * torch.cos(angle)[mask]
        # action[mask, 9] = 0.15 * torch.sin(angle)[mask]
        # action[mask, 5] = -0.9

        return (
            value,
            action,
            action_log_probs,
            rnn_hidden_states,
        )

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info=None,
        evaluate_aux_losses=True,
    ):
        if self.action_distribution_type == "mixed":
            action = action[:, :10]
            if self.action_config.discrete_base:
                action[:, 8:10] = (
                    torch.bucketize(action[:, 8:10], self.boundaries) - 1
                )
            if self.action_config.discrete_arm:
                action[:, :7] = (
                    torch.bucketize(action[:, :7], self.boundaries) - 1
                )
            action[:, 7] = (
                (action[:, 7] == 0).int()
                + 2 * (action[:, 7] == -1).int()
                + 3 * (action[:, 7] == 1).int()
                - 1
            )
        return super().evaluate_actions(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            action,
            rnn_build_seq_info=rnn_build_seq_info,
        )

    def forward(
        self,
        states,
        actions,
        targets,
        rtgs,
        timesteps,
    ):
        if not self.offline_training:
            raise ValueError
        features, planner_logits = self.net(
            states, None, actions, None, rtgs=rtgs, offline_training=True
        )
        # if we are given some desired targets also calculate the loss
        loss = 0
        loss_dict = dict()

        if self.train_planner:
            B = actions.shape[0]

            aux_logits = planner_logits[1]
            planner_logits = planner_logits[0]

            temp_target = torch.cat(
                [
                    states["obj_start_gps_compass"].reshape(B, -1, 2),
                    states["obj_goal_gps_compass"].reshape(B, -1, 2),
                ],
                dim=-1,
            )

            loss_aux = F.mse_loss(aux_logits[..., :4], temp_target)
            loss = loss + loss_aux
            loss_dict.update(
                {
                    "aux_loss": loss_aux.detach().item(),
                }
            )

            if "all_predicates" in states.keys():
                mask_open_skill = (
                    (states["skill"].reshape(B, -1) == 3)
                    | (states["skill"].reshape(B, -1) == 1)
                    | (states["skill"].reshape(B, -1) == 5)
                )
                mask_predicate = mask_open_skill & torch.any(
                    states["all_predicates"].reshape(
                        B, -1, states["all_predicates"].shape[-1]
                    )[..., :5],
                    dim=-1,
                )
                # loss_aux = F.mse_loss(aux_logits[..., 6:9], states["obj_start_sensor"].reshape(B, -1, 3))
                # loss = loss + loss_aux
                # loss_dict.update(
                #     {
                #         "aux_loss_obj": loss_aux.detach().item(),
                #     }
                # )
                temp_target = states["all_predicates"].reshape(
                    B, -1, states["all_predicates"].shape[-1]
                )[..., :5]
                # temp_target = torch.any(temp_target, dim=-1)
                temp_target = torch.cat(
                    [
                        ~torch.any(temp_target, dim=-1, keepdim=True),
                        temp_target,
                    ],
                    dim=-1,
                )
                temp_target = torch.argmax(temp_target.long(), dim=-1)
                # loss_aux = F.mse_loss(aux_logits[..., 4:10], temp_target)
                loss_aux = F.cross_entropy(
                    aux_logits[..., 4:10].permute(0, 2, 1),
                    temp_target.long(),
                    label_smoothing=0.05,
                    reduction="none",
                )
                loss_aux = loss_aux.reshape(B, -1)
                loss_aux = (
                    torch.mean(loss_aux[mask_open_skill])
                    if loss_aux[mask_open_skill].shape[0] != 0
                    else torch.tensor(0)
                )
                loss = loss + loss_aux# * 10
                # accuracy_p = torch.sum(
                #     (torch.argmax(planner_logits[:, :, :6], dim=-1) == 3) &
                #     temp_target
                # ) / torch.sum(temp_target) if torch.sum(temp_target) != 0 else torch.tensor(0)
                accuracy_aux = torch.sum(
                    torch.argmax(
                        aux_logits[mask_open_skill][..., 4:10], dim=-1
                    )
                    == temp_target[mask_open_skill].long()
                ) / np.prod(temp_target[mask_open_skill].shape)
                loss_dict.update(
                    {
                        "aux_loss_2": loss_aux.detach().item(),
                        "accuracy_aux_loss_2": accuracy_aux.detach().item(),
                    }
                )

            # ========================== planner action ============================
            # temp_target = torch.clone(states["skill"].reshape(B, -1, 1))
            temp_target = states["skill"].reshape(B, -1, 1)

            # temp_target[temp_target == 3] = 1
            # temp_target[temp_target == 5] = 1

            loss_p = self.focal_loss_planner(
                planner_logits[:, :, :6].permute(0, 2, 1),
                temp_target.reshape(B, -1).long(),
                # label_smoothing=0.05,
            )
            accuracy_p = torch.sum(
                torch.argmax(planner_logits[:, :, :6], dim=-1)
                == temp_target.reshape(B, -1).long()
            ) / np.prod(temp_target.shape)

            loss_dict.update(
                {
                    "planner_skill": loss_p.detach().item(),
                    "accuracy_planner_skill": accuracy_p.detach().item(),
                }
            )

            if "all_predicates" in states.keys():
                temp_target = (states["skill"].reshape(B, -1) == 5).long()
                # temp_target = (
                #     states["skill_change"].reshape(B, -1) == 1
                # ).long()
                loss_p_2 = self.focal_loss_planner_2(
                    planner_logits[:, :, 6:8].permute(0, 2, 1),
                    temp_target,
                    # label_smoothing=0.05,
                    # reduction='none',
                )
                # loss_p_2 = loss_p_2.reshape(B, -1)
                # loss_p_2 = torch.mean(loss_p_2[mask_predicate]) if loss_p_2[mask_predicate].shape[0] != 0 else torch.tensor(0)
                # loss = loss + loss_p_2 * 10
                # accuracy_p = torch.sum(
                #     (torch.argmax(planner_logits[mask_predicate][:,6:8], dim=-1) == temp_target[mask_predicate])
                # ) / np.prod(temp_target[mask_predicate].shape)
                # loss_p_2 = loss_p_2.reshape(B, -1)
                # loss_p_2 = torch.mean(loss_p_2[mask_open_skill]) if loss_p_2[mask_open_skill].shape[0] != 0 else torch.tensor(0)
                loss = loss + loss_p_2
                accuracy_p = torch.sum(
                    (
                        torch.argmax(
                            planner_logits[mask_open_skill][:, 6:8], dim=-1
                        )
                        == temp_target[mask_open_skill]
                    )
                ) / np.prod(temp_target[mask_open_skill].shape)
                # accuracy_p = torch.sum(
                #     (torch.argmax(planner_logits[:, :, :6], dim=-1) == 3) &
                #     temp_target
                # ) / torch.sum(temp_target) if torch.sum(temp_target) != 0 else torch.tensor(0)
                loss_dict.update(
                    {
                        "accuracy_skill_3": accuracy_p.detach().item(),
                        "loss_skill_3": loss_p_2.detach().item(),
                    }
                )
            loss = loss + loss_p

        if self.train_control:

            if self.action_distribution_type == "categorical":
                distribution = self.action_distribution(features)
                logits = distribution.probs
            elif self.action_distribution_type == "gaussian":
                distribution = self.action_distribution(features)
                logits = distribution.mean
            elif self.action_distribution_type == "mixed":
                logits = self.action_distribution(features, return_logits=True)
            else:
                raise NotImplementedError

            # ======================== separate logits ==========================
            logits_arm, logits_pick, logits_loc = torch.split(
                logits, self.len_logit, -1
            )

            # =========================== locomotion ============================
            if self.action_config.discrete_base:
                temp_target = (
                    torch.bucketize(targets[:, :, 8:10], self.boundaries) - 1
                )

                # temp_target = (temp_target[:,:,0] * 21 + temp_target[:,:,1]).view(*logits_loc.shape[:2], 1).long()

                # logits_loc = logits_loc.view(*logits_loc.shape[:2], 1, 21 * 21)
                logits_loc = logits_loc.view(*logits_loc.shape[:2], 2, 21)
                loss1 = self.focal_loss_loc(
                    logits_loc[:, :, :, :].permute(0, 3, 1, 2),
                    temp_target[:, :, :],
                )
                accuracy1 = torch.sum(
                    torch.argmax(logits_loc[:, :, :, :], dim=-1)
                    == temp_target[:, :, :]
                ) / np.prod(temp_target[:, :, :].shape)
            else:
                temp_target = targets[:, :, 8:10]
                loss1 = F.mse_loss(logits_loc, temp_target)

            # =========================== arm action ============================
            if self.action_config.discrete_arm:
                temp_target = (
                    torch.bucketize(targets[:, :, :7], self.boundaries) - 1
                )
                logits_arm = logits_arm.view(*logits_arm.shape[:2], 7, 21)
                loss2 = self.focal_loss(
                    logits_arm[:, :, :, :].permute(0, 3, 1, 2),
                    temp_target[:, :, :7],
                )
                accuracy2 = torch.sum(
                    torch.argmax(logits_arm[:, :, :, :], dim=-1)
                    == temp_target[:, :, :7]
                ) / np.prod(temp_target[:, :, :7].shape)
            else:
                temp_target = targets[:, :, :7]
                loss2 = F.mse_loss(logits_arm, temp_target)

            # ========================= gripper action ==========================
            loss3 = self.focal_loss_pick(
                logits_pick.permute(0, 2, 1), targets[:, :, 10].long()
            )
            accuracy3 = torch.sum(
                torch.argmax(logits_pick[:, :, :], dim=-1)
                == targets[:, :, 10].long()
            ) / np.prod(targets[:, :, 10].shape)

            # ========================== stop action ============================
            # loss4 = F.cross_entropy(logits_stop.permute(0,2,1), targets[:,:,10].long(), label_smoothing=0.05)
            # accuracy4 = torch.sum(torch.argmax(logits_stop[:,:,:], dim=-1) == targets[:,:,10].long()) / np.prod(targets[:,:,10].shape)

            loss_dict.update(
                {
                    "locomotion": loss1.detach().item(),
                    "arm": loss2.detach().item(),
                    "pick": loss3.detach().item(),
                    # "place": loss4.detach().item(),
                    "accuracy_pick": accuracy3.detach().item(),
                    # "accuracy_place": accuracy4.detach().item(),
                }
            )

            if self.action_config.discrete_base:
                loss_dict.update(
                    {
                        "accuracy_nav": accuracy1.detach().item(),
                    }
                )
            else:
                loss_dict.update(
                    {
                        "mse_base": loss1.detach().item(),
                    }
                )
            if self.action_config.discrete_arm:
                loss_dict.update(
                    {
                        "accuracy_arm": accuracy2.detach().item(),
                    }
                )
            else:
                loss_dict.update(
                    {
                        "mse_arm": loss2.detach().item(),
                    }
                )

            loss1 = torch.exp(-self.loss_vars[0]) * loss1 + self.loss_vars[0]
            loss2 = torch.exp(-self.loss_vars[1]) * loss2 + self.loss_vars[1]
            loss3 = torch.exp(-self.loss_vars[2]) * loss3 + self.loss_vars[2]
            # loss = torch.exp(-self.loss_vars[3]) * loss + self.loss_vars[3]
            loss = loss + loss1 + loss2 + loss3  # + loss4
        return loss, loss_dict

    def get_policy_info(self, infos, dones):
        policy_infos = []
        for i, info in enumerate(infos):
            policy_info = {
                "cur_skill": self.net.cur_skill_all[i],
                "reset_arm": self.reset_mask[i],
                "gripper": self.gripper_action[i],
                # "predicted_dist": "{}, {}; {}, {}".format(
                #     self.net.predicted_dist[i, 0],
                #     self.net.predicted_dist[i, 1],
                #     self.net.predicted_dist[i, 2],
                #     self.net.predicted_dist[i, 3],
                # ),
                "predicted_switch?": "{}".format(
                    self.net.switched[i]
                ),
                "predicted_close??": "{}".format(
                    torch.argmax(self.net.predicted_dist[i, 4:10])
                ),
                "predicted_skill3???": "{}, {}".format(
                    self.net.predicted_skill3[i, 0],
                    self.net.predicted_skill3[i, 1],
                ),
                "timeout": self.net.timeout[i],
                "timeout2": self.net.timeout2[i],
                "timeout3": self.net.timeout3[i],
                # "angle": "{}, {}, {}, {}".format( self.angle[i], 0.2 * torch.cos(self.angle[i]), 0.2 * torch.sin(self.angle[i]), (self.net.cur_skill == 5) & (self.net.timeout > 50)[i])
            }
            policy_infos.append(policy_info)

        return policy_infos

    @property
    def hidden_state_hxs_dim(self):
        return self.net.hidden_state_hxs_dim

    def _reset_arm(
        self, observations, prev_actions, rnn_hidden_states, reset_mask
    ):
        self._target = torch.tensor(
            [
                -4.5003259e-01,
                -1.0799699e00,
                9.9526465e-02,
                9.3869519e-01,
                -7.8854430e-04,
                1.5702540e00,
                4.6168058e-03,
            ],
            device=rnn_hidden_states.device,
        )
        self._initial_delta[reset_mask] = (
            self._target - observations["joint"]
        )[reset_mask]

        current_joint_pos = observations["joint"]
        delta = self._target - current_joint_pos

        # Dividing by max initial delta means that the action will
        # always in [-1,1] and has the benefit of reducing the delta
        # amount was we converge to the target.
        delta = delta / torch.maximum(
            self._initial_delta.max(-1, keepdims=True)[0],
            torch.tensor(1e-5, device=rnn_hidden_states.device),
        )

        prev_actions[self.reset_mask, :7] = delta[self.reset_mask]
        # prev_actions[self.reset_mask, 8:10] = 0

        self.net.timeout[self.reset_mask] -= 1

        self.reset_mask = self.reset_mask & ~(
            torch.abs(current_joint_pos - self._target).max(-1)[0] < 5e-2
        )

    def _back_up(
        self, observations, prev_actions, rnn_hidden_states, reset_mask
    ):
        mask = (self.net.timeout3 > 10) & (self.net.cur_skill == 6)
        prev_actions[mask, 8] = -1
        prev_actions[mask, 9] = 0
        self.net.timeout3[mask] -= 10


class TransformerResnetNet(nn.Module):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        context_length: int,
        max_episode_step: int,
        model_type: str,
        n_head: int,
        n_layer: int,
        reg_flags,
        backbone,
        resnet_baseplanes,
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        fuse_keys: Optional[List[str]] = None,
        include_visual_keys: Optional[List[str]] = None,
        use_rgb=False,
        num_skills=10,
    ):
        super().__init__()
        self.context_length = context_length
        context_length = 30

        self.discrete_actions = discrete_actions
        if discrete_actions:
            num_actions = action_space.n + 1
        else:
            num_actions = get_num_actions(action_space)

        # self._n_prev_action = 32
        # rnn_input_size = self._n_prev_action
        rnn_input_size = 0
        self.include_visual_keys = include_visual_keys

        self._fuse_keys = fuse_keys
        if self._fuse_keys is not None:
            rnn_input_size += sum(
                [observation_space.spaces[k].shape[0] for k in self._fuse_keys]
            )

        self._hidden_size = hidden_size

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        elif (
            self.include_visual_keys is not None
            and len(self.include_visual_keys) != 0
        ):
            use_obs_space = spaces.Dict(
                {
                    k: v
                    for k, v in observation_space.spaces.items()
                    if k in ["robot_head_rgb"]
                }
            )
        else:
            use_obs_space = observation_space

        if use_rgb:
            self.visual_encoder_rgb = ResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                use_input_norm=False,
            )
        else:
            self.visual_encoder_rgb = None

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        elif (
            self.include_visual_keys is not None
            and len(self.include_visual_keys) != 0
        ):
            use_obs_space = spaces.Dict(
                {
                    k: v
                    for k, v in observation_space.spaces.items()
                    if k in ["robot_head_depth"]
                }
            )
        else:
            use_obs_space = observation_space

        self.visual_encoder = ResNetEncoder(
            use_obs_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            use_input_norm=False,
        )

        if not self.visual_encoder.is_blind:
            if use_rgb:
                self.visual_fc_rgb = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder_rgb.output_shape),
                        hidden_size // 2,
                    ),
                    nn.ReLU(True),
                )
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )
            else:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape),
                        hidden_size // 2,
                    ),
                    nn.ReLU(True),
                )
        self._hxs_dim = (self._hidden_size // 2) + rnn_input_size + num_actions
        self._hxs_dim += 1 if num_skills != 0 else 0
        self._num_actions = num_actions
        self.action_dim = self._num_actions
        self.obs_dim = (
            self._num_actions + 1 if num_skills != 0 else self._num_actions
        )
        mconf = GPTConfig(
            num_actions,
            context_length,
            num_states=[
                (0 if self.is_blind else self._hidden_size // 2),
                rnn_input_size,
            ],
            n_layer=n_layer,
            n_head=n_head,
            n_embd=self._hidden_size,
            model_type=model_type,
            max_timestep=max_episode_step,
            num_skills=num_skills,
            use_rgb=use_rgb,
            reg_flags=reg_flags,
        )  # 6,8
        self.state_encoder = GPT(mconf)
        # self.state_encoder = LSTMBC(mconf)
        # self.state_encoder = build_rnn_state_encoder(
        #     256+21 +12,
        #     self._hidden_size,
        #     rnn_type='lstm',
        #     num_layers=2,
        # )

        mconf = GPTConfig(
            num_actions,
            context_length,
            num_states=[
                (0 if self.is_blind else self._hidden_size // 2),
                rnn_input_size,
            ],
            n_layer=2,
            n_head=8,
            n_embd=self._hidden_size,
            model_type=model_type,
            max_timestep=max_episode_step,
            num_skills=num_skills,
            use_rgb=use_rgb,
            reg_flags=reg_flags,
        )  # 6,8
        self.planner_encoder = PlannerGPT(mconf)

        self.train()

    @property
    def hidden_state_hxs_dim(self):
        return self._hxs_dim

    @property
    def num_recurrent_layers(self):
        return self.context_length

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info=None,
        # targets=None,
        rtgs=None,
        # timesteps=None,
        offline_training=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = []
        B = prev_actions.shape[0]

        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                # visual_feats = self.visual_encoder_rgb(observations)
                # visual_feats = self.visual_fc_rgb(visual_feats)
                # x.append(visual_feats)
                visual_feats = self.visual_encoder(observations)
                visual_feats = self.visual_fc(visual_feats)
                x.append(visual_feats)
                if self.visual_encoder_rgb is not None:
                    # if observations['robot_head_rgb'].shape[-2] != self.image_size:
                    #     observations['robot_head_rgb'] = torchvision.transforms.functional.resize( \
                    #                     observations['robot_head_rgb'].permute(0,3,1,2), self.image_size).permute(0,2,3,1)
                    visual_feats = self.visual_encoder_rgb(observations)
                    visual_feats = self.visual_fc_rgb(visual_feats)
                    x.append(visual_feats)

        if self._fuse_keys is not None:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys], dim=-1
            )
            x.append(fuse_states)

        x = torch.cat(x, dim=1)
        x = x.reshape(B, -1, *x.shape[1:])

        if offline_training:
            # Move valid state-action-reward pair to the left
            out2 = self.planner_encoder(x)
            if "skill_control" in observations.keys():
                x = torch.cat(
                    [x, observations["skill_control"].reshape(B, -1, 1).float()],
                    dim=-1,
                )
            elif "skill" in observations.keys():
                x = torch.cat(
                    [x, observations["skill"].reshape(B, -1, 1).float()],
                    dim=-1,
                )
            out = self.state_encoder(
                x,
                prev_actions,
                rtgs=rtgs,
            )
            return out, out2

        rnn_hidden_states *= masks.view(-1, 1, 1)

        current_context = torch.argmax(
            (rnn_hidden_states.sum(-1) == 0).float(), -1
        )

        # Write obs to context
        # print(
        #     f"Rnn shape {rnn_hidden_states.shape}, batch {B}, ctx {current_context}, ac dim {action_dim} x shape {x.shape}, actions shape {prev_actions.shape}"
        # )

        obs_dim = self.obs_dim
        action_dim = self.action_dim

        rnn_hidden_states[
            torch.arange(B), current_context, :-obs_dim
        ] = x.view(B, -1)

        # Write actions to context
        rnn_hidden_states[
            torch.arange(B), current_context, -action_dim:
        ] = prev_actions.view(B, -1)

        out, predicted_dist = self.planner_encoder(
            rnn_hidden_states[..., :-obs_dim],
        )
        self.predicted_dist = predicted_dist[torch.arange(B), current_context]
        self.predicted_skill3 = torch.softmax(
            out[torch.arange(B), current_context, 6:8], dim=-1
        )
        rnn_hidden_states[
            torch.arange(B), current_context, -obs_dim
        ] = torch.argmax(
            out[torch.arange(B), current_context, :6], dim=-1
        ).float()

        if not hasattr(self, "cur_skill"):
            self.cur_skill = torch.zeros(B, device=rnn_hidden_states.device)

        if not hasattr(self, "timeout"):
            self.timeout = torch.zeros(B, device=rnn_hidden_states.device)
            self.timeout2 = torch.zeros(B, device=rnn_hidden_states.device)
            self.timeout3 = torch.zeros(B, device=rnn_hidden_states.device)

        if not hasattr(self, "switched"):
            self.switched = torch.zeros(B, device=rnn_hidden_states.device).bool()

        self.timeout *= masks.view(-1)
        self.timeout2 *= masks.view(-1)
        self.timeout3 *= masks.view(-1)
        self.switched *= masks.view(-1)

        mask = (observations["is_holding"].reshape(-1) == 0) & (rnn_hidden_states[torch.arange(B), current_context, -obs_dim] == 2)
        rnn_hidden_states[torch.arange(B), current_context, -obs_dim] -= mask * 1
        # mask = (observations["is_holding"].reshape(-1) == 1) & (rnn_hidden_states[torch.arange(B), current_context, -obs_dim] == 3)
        # rnn_hidden_states[torch.arange(B), current_context, -obs_dim] -= mask * 1
        
        # HACK
        mask = (
            (observations["obj_start_gps_compass"][:, 0] < 1.5) 
            & (torch.argmax(self.predicted_dist[:,4:10], dim=-1) > 0) 
            & (torch.argmax(self.predicted_dist[:,4:10], dim=-1) < 5)
        )
        self.switched = self.switched | mask
        self.timeout2[mask] = 0

        # mask_place = observations["is_holding"].reshape(-1) == 1
        # mask = rnn_hidden_states[:, current_context, -obs_dim] == 3
        # rnn_hidden_states[:, current_context, -obs_dim] -= 1 * mask_place * mask
        # mask = rnn_hidden_states[:, current_context, -obs_dim] == 5
        # rnn_hidden_states[:, current_context, -obs_dim] -= 3 * mask_place * mask

        mask = rnn_hidden_states[:, current_context, -obs_dim] == 1
        rnn_hidden_states[:, current_context, -obs_dim] += 2 * mask * self.switched 
        # mask = rnn_hidden_states[:, current_context, -obs_dim] == 5
        # rnn_hidden_states[:, current_context, -obs_dim] -= 2 * mask
        # mask = (
        #     (observations["obj_start_gps_compass"][:, 0] < 1.5) 
        #     & (torch.argmax(self.predicted_dist[:,4:10], dim=-1) == 5)
        #     & (rnn_hidden_states[:, current_context, -obs_dim] == 1)
        # )
        # rnn_hidden_states[:, current_context, -obs_dim] += 4 * mask * (self.predicted_skill3[:,1] > 0.5)
        # mask = (
        #     (observations["obj_start_gps_compass"][:, 0] < 1.5) 
        #     & (torch.argmax(self.predicted_dist[:,4:10], dim=-1) == 5)
        #     & (rnn_hidden_states[:, current_context, -obs_dim] == 5)
        # )
        # rnn_hidden_states[:, current_context, -obs_dim] -= 4 * mask * (self.predicted_skill3[:,1] < 0.5)
        mask = (
            (observations["obj_start_gps_compass"][:, 0] < 1.5) 
            # & (torch.argmax(self.predicted_dist[:,4:10], dim=-1) > 0) 
            # & (torch.argmax(self.predicted_dist[:,4:10], dim=-1) < 5)
            & (rnn_hidden_states[:, current_context, -obs_dim] == 3)
        )
        rnn_hidden_states[:, current_context, -obs_dim] += 2 * mask * (self.predicted_skill3[:,1] > 0.1) * self.switched * (self.timeout > 200)
        # mask = (
        #     (observations["obj_start_gps_compass"][:, 0] < 1.5) 
        #     & (torch.argmax(self.predicted_dist[:,4:10], dim=-1) > 0) 
        #     & (torch.argmax(self.predicted_dist[:,4:10], dim=-1) < 5)
        #     & (rnn_hidden_states[:, current_context, -obs_dim] == 5)
        # )
        # rnn_hidden_states[:, current_context, -obs_dim] -= 2 * mask * (self.predicted_skill3[:,1] < 0.25)
        mask = (
            rnn_hidden_states[torch.arange(B), current_context, -obs_dim]
            == self.cur_skill
        ) | (self.cur_skill == 6) | ((self.cur_skill == 5) & (rnn_hidden_states[torch.arange(B), current_context, -obs_dim] == 3)) | (
                (self.cur_skill == 3) & (rnn_hidden_states[torch.arange(B), current_context, -obs_dim] == 5)
            )
        self.timeout[mask] += 1
        self.timeout[~mask] = 0
        self.timeout2[self.switched] += 1
        

        # if not hasattr(self, "last_dist"):
        #     self.last_dist = observations["obj_start_gps_compass"][:, 0]
        # mask = (torch.abs(observations["obj_start_gps_compass"][:, 0] - self.last_dist) < 0.02) & ((rnn_hidden_states[torch.arange(B), current_context, -obs_dim] == 0) | (rnn_hidden_states[torch.arange(B), current_context, -obs_dim] == 4))
        # self.timeout3[mask] += 1
        # self.last_dist = observations["obj_start_gps_compass"][:, 0]
        # mask = self.timeout3 > 100
        # rnn_hidden_states[torch.arange(B), current_context, -obs_dim] = 6 * mask + rnn_hidden_states[torch.arange(B), current_context, -obs_dim] * (~mask)
        # mask = (self.timeout3 > 10) & (self.cur_skill == 6)
        # rnn_hidden_states[
        #     torch.arange(B), current_context, -obs_dim
        # ] = (~mask) * rnn_hidden_states[
        #     torch.arange(B), current_context, -obs_dim
        # ] + 6 * mask


        # if (self.timeout[0] > 350) or (self.timeout[0] > 250 and not (
        #     self.cur_skill[0] == 3
        #     or self.cur_skill[0] == 5
        # )):
        #     prob = (
        #         torch.softmax(out[torch.arange(B), current_context], dim=-1)
        #         .cpu()
        #         .numpy()
        #     )
        #     prob[0, :4] = 0.25
        #     prob[0, 4:] = 0.
        #     select = (
        #         torch.from_numpy(
        #             np.array([np.random.choice(np.arange(10), p=prob[0])])
        #         )
        #         .float()
        #         .cuda()
        #     )
        #     rnn_hidden_states[
        #         torch.arange(B), current_context, -obs_dim
        #     ] = select

        self.reset_mask = (
            (
                rnn_hidden_states[torch.arange(B), current_context, -obs_dim]
                != self.cur_skill
            )
            & (self.cur_skill != 5)
            & (
                rnn_hidden_states[torch.arange(B), current_context, -obs_dim]
                != 5
            )
        )

        mask = (self.timeout > 350) | ((self.timeout > 250) & ~((self.cur_skill == 3) | (self.cur_skill[0] == 5)))
        self.reset_mask[mask] = True
        self.timeout[mask] = 0

            
        mask = self.timeout2 > 250
        self.switched[mask] = False
        self.timeout2[mask] = 0

        self.cur_skill = rnn_hidden_states[
            torch.arange(B), current_context, -obs_dim
        ]

        # rnn_hidden_states[mask, current_context, -obs_dim] += (
        #     2
        #     * (
        #         (self.predicted_skill3[:, 1] > 0.5)
        #         & (observations["obj_start_gps_compass"][:, 0] < 2)
        #         & torch.argmax(self.predicted_dist[:,4:10], dim=-1).bool()
        #     ).float()
        # )
        # torch.argmax(self.predicted_skill3, dim=-1)
        # rnn_hidden_states[
        #     mask, current_context, -obs_dim
        # ] += 2 * (
        #         torch.from_numpy(np.array([np.random.choice(np.arange(2), p=self.predicted_skill3.cpu().numpy()[0])])).cuda()
        #          & (observations["obj_start_gps_compass"][:,0] < 2)
        #     ).float()
        # 2 * torch.argmax(out[torch.arange(B), current_context, 6:8], dim=-1).float()

        self.cur_skill_all = rnn_hidden_states[
            torch.arange(B), current_context, -obs_dim
        ]

        rnn_hidden_states = rnn_hidden_states.contiguous()

        out = self.state_encoder(
            rnn_hidden_states[..., :-action_dim],
            rnn_hidden_states[..., -action_dim:],
            rtgs=None,
        )

        return out[torch.arange(B), current_context], rnn_hidden_states, {}
