import torch
from habitat_baselines.utils.common import (
    iterate_action_space_recursively,
    CustomNormal,
    CustomFixedCategorical,
)
import gym.spaces as spaces


class _SumTensors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *tensors):
        ctx.num_inputs = len(tensors)

        return torch.stack(tensors, -1).sum(-1)

    @staticmethod
    def backward(ctx, grad_out):
        return tuple(grad_out for _ in range(ctx.num_inputs))


def sum_tensor_list(tensors):
    if len(tensors) == 1:
        return tensors[0]
    elif len(tensors) == 2:
        return tensors[0] + tensors[1]
    else:
        return _SumTensors.apply(*tensors)


class ActionDistribution:
    def __init__(self, action_space, box_mu_act, logits, std):
        if std is None:
            self.params = logits
        else:
            self.params = torch.cat((logits, std), -1)

        self.distributions = []
        self.action_slices = []
        self.action_dtypes = []
        logits_offset = 0
        std_offset = 0
        action_offset = 0
        self.dtype = torch.int64
        for space in iterate_action_space_recursively(action_space):
            if isinstance(space, spaces.Box):
                numel = int(np.prod(space.shape))
                mu = logits[..., logits_offset : logits_offset + numel]
                if box_mu_act == "tanh":
                    mu = torch.tanh(mu)
                self.distributions.append(
                    CustomNormal(mu, std[..., std_offset : std_offset + numel])
                )
                std_offset += numel

                self.action_slices.append(
                    slice(action_offset, action_offset + numel)
                )
                self.dtype = torch.float32
                self.action_dtypes.append(torch.float32)
            elif isinstance(space, spaces.Discrete):
                numel = space.n
                self.distributions.append(
                    CustomFixedCategorical(
                        logits=logits[
                            ..., logits_offset : logits_offset + numel
                        ]
                    )
                )
                self.action_slices.append(
                    slice(action_offset, action_offset + 1)
                )
                self.action_dtypes.append(torch.int64)

            logits_offset += numel
            action_offset = self.action_slices[-1].stop

    def sample(self, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size()
        return torch.cat(
            [
                dist.sample(sample_shape).to(self.dtype)
                for dist in self.distributions
            ],
            -1,
        )

    def log_probs(self, action):
        all_log_probs = []
        for dist, _slice, dtype in zip(
            self.distributions, self.action_slices, self.action_dtypes
        ):
            all_log_probs.append(dist.log_probs(action[..., _slice].to(dtype)))

        return sum_tensor_list(all_log_probs)

    def entropy(self):
        return sum_tensor_list([dist.entropy() for dist in self.distributions])
