import math
from math import sqrt
from transformers import Qwen2Model, GenerationMixin, Qwen2ForCausalLM, Qwen2Config
import torch.nn as nn
from typing import Optional, List, Tuple, Union, Dict, Any
import torch
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import QWEN2_INPUTS_DOCSTRING
from transformers.utils import logging, add_start_docstrings_to_model_forward, can_return_tuple
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)

class RMSNorm(torch.nn.Module):
    """Saner dtype handling and slightly better for fusion"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type if x.device.type != "meta" else "cuda"):
            return self._norm(x.float()).type_as(x) * self.weight

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


class RecurrentQwenConfig(Qwen2Config):
    def __init__(self,
                 prelude_layer=12,
                 coda_layer=12,
                 mean_recurrence=1,
                 mean_backprop_depth=4,
                 **kwargs):
        self.prelude_layer = prelude_layer
        self.coda_layer = coda_layer
        self.mean_recurrence = mean_recurrence
        self.mean_backprop_depth = mean_backprop_depth
        super().__init__(**kwargs)


class RecurrentDynamicCache(DynamicCache):
    # By default, the cache is compressed to the last recurrent state (skip the middle recurrent states)
    def __init__(self):
        super().__init__()
        self.key_cache: dict[int, torch.Tensor] = {}
        self.value_cache: dict[int, torch.Tensor] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(layer_idx, int):
            raise ValueError(
                "layer_idx must be an integer but is", type(layer_idx)
            )

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache[layer_idx] = key_states.permute(2, 0, 1, 3)
            self.value_cache[layer_idx] = value_states.permute(2, 0, 1, 3)
        else:
        # the only difference with the original implementation is that we cannot concatenate
        # the old cache with key_states and value_states, as we may update the same position
        # multiple times in a single forward pass
            if len(self.key_cache[layer_idx]) == self._seen_tokens:
                #if key_states.shape[-2] == self._seen_tokens: # special case for training (gradient mode)
                #    self.key_cache[layer_idx] = key_states.permute(2, 0, 1, 3)
                #    self.value_cache[layer_idx] = value_states.permute(2, 0, 1, 3)
                #else:
                for idx, entry in enumerate(key_states.unbind(dim=-2)):
                    self.key_cache[layer_idx][self._seen_tokens - key_states.shape[-2] + idx] = entry
                for idx, entry in enumerate(value_states.unbind(dim=-2)):
                    self.value_cache[layer_idx][self._seen_tokens - value_states.shape[-2] + idx] = entry
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states.permute(2, 0, 1, 3)], dim=0)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states.permute(2, 0, 1, 3)], dim=0)

        return (
            self.key_cache[layer_idx].permute(1, 2, 0, 3),
            self.value_cache[layer_idx].permute(1, 2, 0, 3),
        )

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "RecurrentDynamicCache":
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.key_cache[layer_idx] = key_states
                cache.value_cache[layer_idx] = value_states

                if layer_idx == 0:
                    cache._seen_tokens = len(key_states)
        return cache

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens


class RecurrentQwen(Qwen2Model):
    config_class = RecurrentQwenConfig

    def __init__(self, config: RecurrentQwenConfig):
        prelude_layer       = getattr(config, "prelude_layer",       12)
        coda_layer          = getattr(config, "coda_layer",          12)
        mean_recurrence     = getattr(config, "mean_recurrence",     1)
        mean_backprop_depth = getattr(config, "mean_backprop_depth", 4)

        print(f"prelude_layer: {prelude_layer}, coda_layer: {coda_layer}, mean_recurrence: {mean_recurrence}, mean_backprop_depth: {mean_backprop_depth}")

        super().__init__(config)
        self.prelude_layer = prelude_layer
        self.coda_layer = coda_layer
        self.core_block_layers = config.num_hidden_layers - prelude_layer - coda_layer
        self.n_embd = config.hidden_size

        self.adapter = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        with torch.no_grad():
            nn.init.zeros_(self.adapter.weight)
            self.adapter.weight.data[:, :config.hidden_size] = torch.eye(config.hidden_size)
            """
            eps = 1.0 / sqrt(self.n_embd)
            noise = torch.randn_like(self.adapter.weight)
            std = sqrt(2 / (5 * self.n_embd))
            torch.nn.init.trunc_normal_(noise, mean=0.0, std=std, a=-3 * std, b=3 * std)
            noise = noise * eps
            self.adapter.weight.add_(noise)
            """

        # self.ln_f = RMSNorm(config.hidden_size, eps=1e-6)

        self.mean_recurrence = mean_recurrence
        self.mean_backprop_depth = mean_backprop_depth

        self.effective_expected_depth = (
            self.prelude_layer + self.core_block_layers * self.mean_recurrence + self.coda_layer
        )
        self.init_values = {
            "std": sqrt(2 / (5 * self.n_embd)),
            "out_proj": sqrt(2 / (5 * self.n_embd)) / sqrt(2 * self.effective_expected_depth),
            "embedding": sqrt(2 / (5 * self.n_embd)),
            "embed_scale": sqrt(self.n_embd),
        }

    def initialize_state(self, input_embeds, deterministic: bool = False):
        x = torch.randn_like(input_embeds)
        std = self.init_values["std"]
        torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if self.init_values["embed_scale"] != 1:
            x = x * self.init_values["embed_scale"]
        return x if not deterministic else x.zero_()

    @torch._dynamo.disable(recursive=False)
    def randomized_iteration_sampler(self) -> tuple[torch.Tensor, torch.Tensor]:
        t = max(self.mean_recurrence - self.mean_backprop_depth, 0)
        s = self.mean_backprop_depth
        if torch.rand((1,)).is_meta:
            return t, s
        if self.training:
            sigma = 0.5
            mu = math.log(t + s) - (sigma**2 / 2)
            rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma)
            p = torch.poisson(torch.tensor([rate], dtype=torch.float)) + 1
            n = torch.clamp(p - s, min=0)
            k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))
            #n, k = torch.as_tensor(0), torch.as_tensor(self.mean_recurrence)
        else:
            n, k = torch.as_tensor(self.mean_recurrence), torch.as_tensor(0)

        return n.to(dtype=torch.long), k.to(dtype=torch.long)

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.training and use_cache:
            use_cache = False

        if not isinstance(past_key_values, (type(None), RecurrentDynamicCache)):
            if past_key_values._seen_tokens == 0:
                past_key_values = RecurrentDynamicCache()
            else:
                raise ValueError("The `past_key_values` should be either a `RecurrentDynamicCache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = RecurrentDynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # 1. Prelude
        for layer_idx in range(self.prelude_layer):
            decoder_layer = self.layers[layer_idx]
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 2. Current Core Blocks
        if self.core_block_layers > 0:
            memory_state, core_hidden_states, core_self_attns = self.process_core_blocks(
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                output_hidden_states,
                position_embeddings,
                **flash_attn_kwargs
            )

            hidden_states = memory_state
            if output_hidden_states and core_hidden_states:
                all_hidden_states += core_hidden_states
            if output_attentions and core_self_attns:
                all_self_attns += core_self_attns

        # 3. Coda
        for layer_idx in range(self.prelude_layer + self.core_block_layers, len(self.layers)):
            decoder_layer = self.layers[layer_idx]
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @torch._dynamo.disable(recursive=False)
    def process_core_blocks(
        self,
        hidden_states,
        causal_mask,
        position_ids,
        past_key_values,
        output_attentions,
        use_cache,
        cache_position,
        output_hidden_states,
        position_embeddings,
        **flash_attn_kwargs
    ):
        all_hidden_states = ()
        all_self_attns = ()

        if self.core_block_layers > 0:
            middle_start = self.prelude_layer
            middle_end = self.prelude_layer + self.core_block_layers
            memory_state = self.initialize_state(hidden_states)

            num_steps_no_grad, num_steps_with_grad = self.randomized_iteration_sampler()

            with torch.no_grad():
                for step in range(num_steps_no_grad):
                    combined_input = torch.cat([hidden_states, memory_state], dim=-1)
                    memory_state = self.adapter(combined_input)

                    for layer_idx in range(middle_start, middle_end):

                        decoder_layer = self.layers[layer_idx]

                        layer_outputs = decoder_layer(
                            memory_state,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            **flash_attn_kwargs
                        )

                        if output_hidden_states:
                            all_hidden_states += (memory_state,)

                        if output_attentions:
                            all_self_attns += (layer_outputs[1],)

                        memory_state = layer_outputs[0]

                # memory_state = self.ln_f(memory_state)

            for step in range(num_steps_with_grad):
                combined_input = torch.cat([hidden_states, memory_state], dim=-1)
                memory_state = self.adapter(combined_input)

                for layer_idx in range(middle_start, middle_end):

                    decoder_layer = self.layers[layer_idx]

                    layer_outputs = decoder_layer(
                        memory_state,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs
                    )

                    if output_hidden_states:
                        all_hidden_states += (memory_state,)

                    if output_attentions:
                        all_self_attns += (layer_outputs[1],)

                    memory_state = layer_outputs[0]
                # memory_state = self.ln_f(memory_state)

        else:
            memory_state = hidden_states

        return memory_state, all_hidden_states, all_self_attns


class RecurrentQwenForCausalLM(Qwen2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = RecurrentQwenConfig

    def __init__(self, config: RecurrentQwenConfig):
        super().__init__(config)
        self.model = RecurrentQwen(config)