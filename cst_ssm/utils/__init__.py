"""Utilities: sharded checkpoints, LoRA, memory, config, distributed."""
from __future__ import annotations

from .checkpoint import (save_sharded, load_sharded, verify_shards, load_checkpoint,
                         trainable_state_dict, frozen_param_names,
                         base_weights_recoverable, checkpoint_metadata)
from .distributed import (maybe_init_distributed, check_ddp_launch, dist_info,
                          resolve_device, launched_distributed, is_main_process,
                          get_rank, get_world_size, add_ddp_args, resolve_ddp,
                          check_device_binding, seed_everything, cleanup_distributed)
from .lora import LoRALinear, apply_lora, apply_qlora, mark_only_lora_trainable
from .memory import (set_eacs_chunk, set_gate_temperature, autocast_ctx,
                     build_optimizer, gate_temperature_schedule,
                     set_eps_anneal, eps_anneal_schedule)
from .config import (load_yaml, save_yaml, model_config_from_dict, branches_from_list,
                     train_config_from_yaml, loss_weights_from_yaml, loader_config_from_yaml,
                     require_model_config)

__all__ = [
    "save_sharded", "load_sharded", "verify_shards", "load_checkpoint",
    "trainable_state_dict", "frozen_param_names", "base_weights_recoverable",
    "checkpoint_metadata",
    "maybe_init_distributed", "check_ddp_launch", "dist_info", "resolve_device",
    "launched_distributed", "is_main_process", "get_rank", "get_world_size",
    "add_ddp_args", "resolve_ddp", "check_device_binding",
    "seed_everything", "cleanup_distributed",
    "LoRALinear", "apply_lora", "apply_qlora", "mark_only_lora_trainable",
    "set_eacs_chunk", "set_gate_temperature", "autocast_ctx",
    "build_optimizer", "gate_temperature_schedule",
    "set_eps_anneal", "eps_anneal_schedule",
    "load_yaml", "save_yaml", "model_config_from_dict", "branches_from_list",
    "train_config_from_yaml", "loss_weights_from_yaml", "loader_config_from_yaml",
    "require_model_config",
]
