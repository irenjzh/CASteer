import os
import numpy as np
import torch
import abc
from collections import defaultdict
from typing import Optional, Union, Tuple, List, Callable, Dict, Any


class VectorControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, vector, layer_idx: int):
        raise NotImplementedError

    def __call__(self, vector, layer_idx: int):
        vector = self.forward(vector, layer_idx)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()
            self.cur_step += 1
        return vector


class SanaVectorStore(VectorControl):
    def __init__(self, steering_vectors=None, steer=True,
                 alpha=10, beta=2,
                 steer_back=False,
                 active_layers=None,
                 timestep_scaling=False,
                 total_steps=20,
                 device='cpu'):
        super().__init__()
        self.step_store = self.get_empty_store()
        self.vector_store = defaultdict(dict)
        self.steering_vectors = steering_vectors
        self.steer = steer
        self.alpha = alpha
        self.beta = beta
        self.steer_back = steer_back
        self.device = device
        # Which layers to apply steering (None = all layers)
        self.active_layers = active_layers
        # Whether to scale alpha by timestep (stronger early, weaker late)
        self.timestep_scaling = timestep_scaling
        self.total_steps = total_steps

    def reset(self):
        super().reset()
        self.step_store = self.get_empty_store()
        self.vector_store = defaultdict(dict)

    @staticmethod
    def get_empty_store():
        return {"layers": []}

    def _get_alpha(self):
        if self.timestep_scaling:
            # Linear decay: alpha * (1 - step/total_steps)
            decay = 1.0 - (self.cur_step / max(self.total_steps, 1))
            return self.alpha * max(decay, 0.1)
        return self.alpha

    def forward(self, vector, layer_idx: int):
        if self.steer and self.steering_vectors is not None:
            # Check if this layer should be steered
            if self.active_layers is None or layer_idx in self.active_layers:
                keys = sorted(self.steering_vectors.keys())
                if len(keys) == 1:
                    num_steer = keys[0]
                else:
                    max_key = keys[-1]
                    num_steer = min(self.cur_step, max_key)

                steering_vector = self.steering_vectors[num_steer]["layers"][layer_idx]
                steering_vector = torch.tensor(steering_vector, dtype=vector.dtype, device=self.device).view(1, 1, -1)

                # Skip if steering vector is NaN or zero
                if not (torch.isnan(steering_vector).any() or steering_vector.abs().sum() == 0):
                    norm = torch.norm(vector, dim=2, keepdim=True)

                    if self.steer_back:
                        sim = torch.tensordot(vector, steering_vector,
                                              dims=([2], [2])).view(vector.size()[0], vector.size()[1], 1)
                        sim = torch.where(sim > 0, sim, 0)
                        vector = vector - (self.beta * sim) * steering_vector.expand(1, vector.size()[1], -1)
                    else:
                        alpha = self._get_alpha()
                        vector = vector + alpha * steering_vector.expand(1, vector.size()[1], -1)

                    # Renormalize
                    vector = vector / torch.norm(vector, dim=2, keepdim=True)
                    vector = vector * norm
# для чего обрабатывать этот случай и нормировать 
        # Save activation for computing steering vectors
        self.step_store["layers"].append(
            vector.data.cpu().numpy()[len(vector) // 2:].mean(axis=0).mean(axis=0)
        )
        return vector

    def between_steps(self):
        self.vector_store[self.cur_step] = self.step_store
        self.step_store = self.get_empty_store()


def register_vector_control_sana(model, controller, hook_point="cross_attn"):
    """
    Register steering vector controller for SANA transformer.

    Args:
        model: pipe.transformer (SanaTransformer2DModel)
        controller: SanaVectorStore instance
        hook_point: "cross_attn" (default), "self_attn", or "residual"
    """
    def block_forward(block, layer_idx):
        original_forward = block.forward

        def forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            height: int = None,
            width: int = None,
        ) -> torch.Tensor:
            batch_size = hidden_states.shape[0]

            # 1. Modulation
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                block.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)

            # 2. Self Attention
            norm_hidden_states = block.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

            attn_output = block.attn1(norm_hidden_states)

            if hook_point == "self_attn":
                attn_output = controller(attn_output, layer_idx)

            hidden_states = hidden_states + gate_msa * attn_output

            # 3. Cross Attention
            if block.attn2 is not None:
                attn_output = block.attn2(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                )
                if hook_point == "cross_attn":
                    attn_output = controller(attn_output, layer_idx)

                hidden_states = attn_output + hidden_states

            if hook_point == "residual":
                hidden_states = controller(hidden_states, layer_idx)

            # 4. Feed-forward
            norm_hidden_states = block.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
            ff_output = block.ff(norm_hidden_states)
            ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
            hidden_states = hidden_states + gate_mlp * ff_output

            return hidden_states

        return forward

    count = 0
    for i, block in enumerate(model.transformer_blocks):
        block.forward = block_forward(block, i)
        count += 1

    controller.num_att_layers = count
    print(f"Registered controller for {count} SanaTransformerBlocks (hook: {hook_point})")
