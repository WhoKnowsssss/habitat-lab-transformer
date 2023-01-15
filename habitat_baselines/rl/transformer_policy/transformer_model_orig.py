import math
import logging
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class ActionNorm(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        assert mean.shape[0] == std.shape[0], "Shape Must Match"
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, y):
        # print(y)
        # x = y * self.std + self.mean
        # print(x)
        return y * self.std + self.mean


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """GPT-1 like network roughly 125M params"""

    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        if config.reg_flags['attention_dropout']:
            self.attn_drop = nn.Dropout(config.attn_pdrop)
            self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x)
            .view(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x)
            .view(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x)
            .view(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        # if attention_mask is not None:
        #     att = att.masked_fill(attention_mask.repeat(T,self.n_head,1,1).transpose(0,2) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        if hasattr(self, 'attn_drop'):
            att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        if hasattr(self, 'resid_drop'):
            y = self.resid_drop(self.proj(y))
        else:
            y = self.proj(y)
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        if config.reg_flags['attention_layernorm']:
            self.ln1 = nn.LayerNorm(config.n_embd)
        if config.reg_flags['feedforward_layernorm']:
            self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        if config.reg_flags['feedforward_dropout']:
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.resid_pdrop),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
            )
        

    def forward(self, x):
        if hasattr(self, 'ln1'):
            x = x + self.attn(self.ln1(x))
        else:
            x = x + self.attn(x)
        if hasattr(self, 'ln2'):
            x = x + self.mlp(self.ln2(x))
        else:
            x = x + self.mlp(x)
        
        return x


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        self.num_inputs = 3

        config.block_size = config.block_size * self.num_inputs

        self.reg_flags = config.reg_flags

        self.block_size = config.block_size

        self.n_embd = config.n_embd
        # input embedding stem
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )
        if config.num_skills != 0: 
            self.skill_embedding = nn.Parameter(
                torch.randn(config.num_skills, config.n_embd)
            )
        if self.reg_flags['outer_dropout']:
            self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )

        # decoder head
        if self.reg_flags['outer_layernorm']:
            self.ln_f = nn.LayerNorm(config.n_embd)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e",
            sum(p.numel() for p in self.parameters()),
        )

        if config.num_states[0] == 0:
            self.state_encoder = nn.Sequential(
                nn.Linear(config.num_states[1], config.n_embd), nn.Tanh()
            )
        else:
            self.state_encoder = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(i, config.n_embd // 2), nn.Tanh())
                    for i in [config.num_states[1]]
                ]
            )

        # self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(
            nn.Linear(config.vocab_size, config.n_embd), nn.Tanh()
        )
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

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, whitelist_weight_modules
                ):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, blacklist_weight_modules
                ):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        if config.num_skills != 0: 
            no_decay.add("skill_embedding")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (
            str(inter_params),
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas,
        )
        return optimizer

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
        # vision_embeddings = self.vision_encoder(visual_input.reshape(-1, 1, 128, 128).type(torch.float32).contiguous())
        # vision_embeddings = vision_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd//2) # (batch, block_size, n_embd)

        for i in range(1, len(state_inputs)):
            state_inputs[i] = self.state_encoder[i - 1](
                state_inputs[i].type(torch.float32)
            )

        if actions is not None and self.model_type == "reward_conditioned":
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            actions = torch.clone(actions)
            actions[:, :, :7] = (
                torch.bucketize(actions[:, :, :7], self.boundaries) - 1
            ) / 10
            actions[:, :, [10]] = 0
            actions = actions.type(torch.float32)
            # targets = torch.bucketize(targets[:,:,:], self.boundaries) - 1
            # if actions.shape[-1] == 12:
            #     actions = torch.cat([actions[:,:,:10], actions[:,:,11:]], dim=-1)
            action_embeddings = self.action_embeddings(
                actions
            )  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (
                    states.shape[0],
                    self.num_inputs * states.shape[1],
                    self.config.n_embd,
                ),
                dtype=torch.float32,
                device=action_embeddings.device,
            )

            token_embeddings[:, :: self.num_inputs, :] = rtg_embeddings

            # for i in range(len(state_inputs)):
            #     token_embeddings[:,(i+1)::self.num_inputs,:] = state_inputs[i]
            # token_embeddings[:,1::self.num_inputs,:] = state_inputs[0]
            # token_embeddings[:,2::self.num_inputs,:] = torch.cat([state_inputs[1], state_inputs[-1]], dim=-1)
            token_embeddings[:, 1 :: self.num_inputs, :] = torch.cat(
                [state_inputs[0], state_inputs[-1]], dim=-1
            )

            token_embeddings[
                :, (self.num_inputs - 1) :: self.num_inputs, :
            ] = action_embeddings

        elif actions is not None and self.model_type == "bc":
            actions = torch.clone(actions)
            # actions[:, :, :7] = (
            #     torch.bucketize(actions[:, :, :7], self.boundaries) - 1
            # ) / 10 * 2 - 1
            actions[:,:,[10]] = 0
            actions = actions.type(torch.float32)
            action_embeddings = self.action_embeddings(
                actions
            )  # (batch, block_size, n_embd)
            if skill_set is not None:
                token_embeddings = torch.zeros(
                    (
                        states.shape[0],
                        (self.num_inputs) * states.shape[1],
                        self.config.n_embd,
                    ),
                    dtype=torch.float32,
                    device=action_embeddings.device,
                )
                token_embeddings[:,  :: (self.num_inputs), :] = self.skill_embedding[skill_set.long()].repeat(1,1,1,1).view(skill_set.shape[0],-1,self.config.n_embd)
                token_embeddings[:, 1 :: (self.num_inputs), :] = torch.cat(
                    [state_inputs[0], state_inputs[-1]], dim=-1
                )

                token_embeddings[
                    :, (self.num_inputs - 1) :: (self.num_inputs), :
                ] = action_embeddings
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        batch_size = states.shape[0]
        # all_global_pos_emb = torch.repeat_interleave(
        #     self.global_pos_emb, batch_size, dim=0
        # )  # batch_size, traj_length, n_embd

        # position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :]
        x = token_embeddings + position_embeddings
        if self.reg_flags['outer_dropout']:
            x = self.drop(x)
        x = self.blocks(x)
        if self.reg_flags['outer_layernorm']:
            x = self.ln_f(x)

        if actions is not None and self.model_type == "reward_conditioned":
            return x[:, (self.num_inputs - 2) :: (self.num_inputs), :]
        elif actions is not None and self.model_type == "bc":
            if skill_set is not None: 
                return x[:, 1 :: (self.num_inputs), :]
            return x[:, (self.num_inputs - 3) :: (self.num_inputs - 1), :]
        else:
            raise NotImplementedError()



    
class PlannerGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        self.num_inputs = 1

        config.block_size = config.block_size * self.num_inputs

        self.block_size = config.block_size

        self.n_embd = config.n_embd
        # input embedding stem
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )

        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e",
            sum(p.numel() for p in self.parameters()),
        )

        if config.num_states[0] == 0:
            self.state_encoder = nn.Sequential(
                nn.Linear(config.num_states[1], config.n_embd), nn.Tanh()
            )
        else:
            self.state_encoder = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(i, config.n_embd // 2), nn.Tanh())
                    for i in [config.num_states[1]]
                ]
            )
        self.output_head = nn.Linear(config.n_embd, config.num_skills)
        self.output_head2 = nn.Linear(config.n_embd, 4)
        self.output_head3 = nn.Linear(config.n_embd, 6)
        # self.output_head2.requires_grad_(False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states):

        state_inputs = list(
            torch.split(
                states, [self.n_embd // 2, self.config.num_states[1]], -1
            )
        )
        
        for i in range(1, len(state_inputs)):
            state_inputs[i] = self.state_encoder[i - 1](
                state_inputs[i].type(torch.float32)#[..., 3:13]
            )

        token_embeddings = torch.zeros(
            (
                states.shape[0],
                states.shape[1],
                self.config.n_embd,
            ),
            dtype=torch.float32,
            device=states.device,
        )

        token_embeddings[:,::self.num_inputs,:] = torch.cat(
            [state_inputs[0], state_inputs[-1]], dim=-1
        )
    
        position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        return self.output_head(x), torch.cat([self.output_head2(x), self.output_head3(x)], dim=-1)
