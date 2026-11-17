import dataclasses
from typing import TypedDict
from transformers.utils import logging
from collections.abc import Callable

from flax.core.frozen_dict import FrozenDict

logger = logging.get_logger(__name__)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class RopeScalingConf(TypedDict):
  type: str
  factor: int
  mscale: float
  beta_fast: int
  beta_slow: int
  mscale_all_dim: float
  original_max_position_embeddings: int


@dataclasses.dataclass(unsafe_hash=True, eq=True)
class Config:
  r"""
  This is the configuration class to store the configuration of a [`DeepseekV3Model`]. It is used to instantiate an DeepSeek
  model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
  defaults will yield a similar configuration to that of the DeepSeek-V3.
  Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
  documentation from [`PretrainedConfig`] for more information.
  Args:
      vocab_size (`int`, *optional*, defaults to 129280):
          Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
          `inputs_ids` passed when calling [`DeepseekV3Model`]
      hidden_size (`int`, *optional*, defaults to 4096):
          Dimension of the hidden representations.
      intermediate_size (`int`, *optional*, defaults to 11008):
          Dimension of the MLP representations.
      moe_intermediate_size (`int`, *optional*, defaults to 1407):
          Dimension of the MoE representations.
      num_hidden_layers (`int`, *optional*, defaults to 32):
          Number of hidden layers in the Transformer decoder.
      num_nextn_predict_layers (`int`, *optional*, defaults to 1):
          Number of nextn predict layers in the DeepSeekV3 Model.
      num_attention_heads (`int`, *optional*, defaults to 32):
          Number of attention heads for each attention layer in the Transformer decoder.
      n_shared_experts (`int`, *optional*, defaults to None):
          Number of shared experts, None means dense model.
      n_routed_experts (`int`, *optional*, defaults to None):
          Number of routed experts, None means dense model.
      routed_scaling_factor (`float`, *optional*, defaults to 1.0):
          Scaling factor or routed experts.
      topk_method (`str`, *optional*, defaults to `gready`):
          Topk method used in routed gate.
      n_group (`int`, *optional*, defaults to None):
          Number of groups for routed experts.
      topk_group (`int`, *optional*, defaults to None):
          Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
      num_experts_per_tok (`int`, *optional*, defaults to None):
          Number of selected experts, None means dense model.
      moe_layer_freq (`int`, *optional*, defaults to 1):
          The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
      first_k_dense_replace (`int`, *optional*, defaults to 0):
          Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                          \--k dense layers--/
      norm_topk_prob (`bool`, *optional*, defaults to False):
          Whether to normalize the weights of the routed experts.
      scoring_func (`str`, *optional*, defaults to 'softmax'):
          Method of computing expert weights.
      aux_loss_alpha (`float`, *optional*, defaults to 0.001):
          Auxiliary loss weight coefficient.
      seq_aux = (`bool`, *optional*, defaults to True):
          Whether to compute the auxiliary loss for each individual sample.
      num_key_value_heads (`int`, *optional*):
          This is the number of key_value heads that should be used to implement Grouped Query Attention. If
          `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
          `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
          converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
          by meanpooling all the original heads within that group. For more details checkout [this
          paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
          `num_attention_heads`.
      hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
          The non-linear activation function (function or string) in the decoder.
      max_position_embeddings (`int`, *optional*, defaults to 2048):
          The maximum sequence length that this model might ever be used with.
      initializer_range (`float`, *optional*, defaults to 0.02):
          The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
      rms_norm_eps (`float`, *optional*, defaults to 1e-06):
          The epsilon used by the rms normalization layers.
      use_cache (`bool`, *optional*, defaults to `True`):
          Whether or not the model should return the last key/values attentions (not used by all models). Only
          relevant if `config.is_decoder=True`.
      pad_token_id (`int`, *optional*):
          Padding token id.
      bos_token_id (`int`, *optional*, defaults to 1):
          Beginning of stream token id.
      eos_token_id (`int`, *optional*, defaults to 2):
          End of stream token id.
      pretraining_tp (`int`, *optional*, defaults to 1):
          Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
          document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
          necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
          issue](https://github.com/pytorch/pytorch/issues/76232).
      tie_word_embeddings (`bool`, *optional*, defaults to `False`):
          Whether to tie weight embeddings
      rope_theta (`float`, *optional*, defaults to 10000.0):
          The base period of the RoPE embeddings.
      rope_scaling (`Dict`, *optional*):
          Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
          strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
          `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
          `max_position_embeddings` to the expected new maximum.
      attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
          Whether to use a bias in the query, key, value and output projection layers during self-attention.
      attention_dropout (`float`, *optional*, defaults to 0.0):
          The dropout ratio for the attention probabilities.
  ```python
  >>> from transformers import DeepseekV3Model, DeepseekV3Config
  >>> # Initializing a Deepseek-V3 style configuration
  >>> configuration = DeepseekV3Config()
  >>> # Accessing the model configuration
  >>> configuration = model.config
  ```"""

  model_type = 'deepseek_v3'
  keys_to_ignore_at_inference: tuple[str, ...] = ('past_key_values',)
  vocab_size: int = 129280
  hidden_size: int = 7168
  intermediate_size: int = 18432
  moe_intermediate_size: int = 2048
  num_hidden_layers: int = 61
  num_nextn_predict_layers: int = 1
  num_attention_heads: int = 128
  num_key_value_heads: int = 128
  n_shared_experts: int = 1
  n_routed_experts: int = 256
  ep_size: int = 1
  routed_scaling_factor: float = 2.5
  kv_lora_rank: int = 512
  q_lora_rank: int | None = 1536
  qk_rope_head_dim: int = 64
  v_head_dim: int = 128
  qk_nope_head_dim: int = 128
  topk_method: str = 'noaux_tc'
  n_group: int = 8
  topk_group: int = 4
  num_experts_per_tok: int = 8
  moe_layer_freq: int = 1
  first_k_dense_replace: int = 3
  norm_topk_prob: bool = True
  scoring_func: str = 'sigmoid'
  aux_loss_alpha: float = 0.001
  seq_aux: bool = True
  hidden_act: str | Callable = 'silu'
  max_position_embeddings: int = 4096
  initializer_range: float = 0.02
  rms_norm_eps: float = 1e-6
  use_cache: bool = True
  pad_token_id: int | None = None
  bos_token_id: int = 0
  eos_token_id: int = 1
  pretraining_tp: int = 1
  tie_word_embeddings: bool = False
  rope_theta: float = 10000.0
  rope_scaling: RopeScalingConf | None = None
  attention_bias: bool = False
  attention_dropout: float = 0.0

  def __post_init__(self):
    # for backward compatibility
    if self.num_key_value_heads is None:
      self.num_key_value_heads = self.num_attention_heads

    if self.rope_scaling is not None:
      self.rope_scaling = FrozenDict(self.rope_scaling)

def get_config():
  return Config()