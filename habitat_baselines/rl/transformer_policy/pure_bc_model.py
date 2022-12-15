import math
import logging
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np

class LSTMBC(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def layer_init(self):
        for name, param in self.blocks.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        self.num_inputs = 3

        config.block_size = config.block_size * self.num_inputs

        self.block_size = config.block_size

        self.n_embd = config.n_embd

        self.blocks = nn.LSTM(256+21+32, self.n_embd, num_layers=4, batch_first=True)

        self.layer_init()

        logger.info(
            "number of parameters: %e",
            sum(p.numel() for p in self.parameters()),
        )

        self.action_embeddings = nn.Sequential(
            nn.Linear(config.vocab_size, 32), nn.Tanh()
        )
        # self.tok_emb = nn.Linear(256+32+21, 512)
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        self.boundaries_mean = torch.linspace(-1, 1, 21 ).cuda()
        self.boundaries = torch.linspace(-1.025, 1.025, 22).cuda()

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # state, action, and return
    def forward(self, states, actions, rtgs=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 8)
        # targets: (batch, block_size, 8)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
        # attention_mask: (batch, block_size)

        # assert (
        #     states.shape[1] == actions.shape[1]
        #     and actions.shape[1] == rtgs.shape[1]
        # ), "Dimension must match, {}, {}, {}".format(
        #     states.shape[1], actions.shape[1], rtgs.shape[1]
        # )
        t1 = time.perf_counter()

        if states.shape[-1] == self.n_embd // 2 + self.config.num_states[1] + 1:
            states, skill_set = torch.split(
                states, [self.n_embd // 2 + self.config.num_states[1], 1], -1
            )
        else:
            skill_set = None

        state_inputs = list(
            torch.split(
                states, [self.n_embd // 2, self.config.num_states[1]], -1
            )
        )
        
        # for i in range(1, len(state_inputs)):
        #     state_inputs[i] = self.state_encoder[i - 1](
        #         state_inputs[i].type(torch.float32)
        #     )

        if actions is not None and self.model_type == "bc":
            # temp_a = actions[:, :, :7].contiguous()
            actions = torch.clone(actions)
            actions = actions.type(torch.float32)
            action_embeddings = self.action_embeddings(
                actions
            )  # (batch, block_size, n_embd)
            
            token_embeddings = torch.cat(
                    [state_inputs[0], state_inputs[-1], action_embeddings], dim=-1
                )

            # token_embeddings = self.tok_emb(token_embeddings)
            
        
        batch_size = states.shape[0]
 
        # position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :]
        
        # x = self.drop(token_embeddings + position_embeddings)
        # x = self.blocks(x)
        x, _ = self.blocks(token_embeddings)
        # x = self.ln_f(x)

        

        return x[:,:,:]
        if actions is not None and self.model_type == 'reward_conditioned':
            return x[:, (self.num_inputs - 2) :: (self.num_inputs), :]
        elif actions is not None and self.model_type == 'bc':
            if skill_set is not None: 
                return x[:, 1 :: (self.num_inputs), :]
            return x[:, (self.num_inputs - 3) :: (self.num_inputs - 1), :]
        else:
            raise NotImplementedError()