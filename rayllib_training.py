#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 13, 15:50:31
@last modified : 2021 Apr 21, 14:45:07
"""

from SeegaEnv import RaySeegaEnv
from basic_agent import AI as BasicAI
from smart_agent import AI as SmartAI
from core.player import Color

from itertools import count

import ray
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print

SAVE_STEP = 1

config = ppo.DEFAULT_CONFIG.copy()
extra_config = {
    "env_config": {"shape": (5, 5), "oponent": SmartAI(Color.green)},
    "vf_clip_param": 12.0,
    "num_workers": 32,
    "num_envs_per_worker": 15,
    "framework":"torch",
    # "tf_session_args": {
        # note: overridden by `local_tf_session_args`
        # "intra_op_parallelism_threads": 2,
        # "inter_op_parallelism_threads": 2,
        # "gpu_options": {"allow_growth": True},
        # "log_device_placement": False,
        # "device_count": {"CPU": 1},
        # "allow_soft_placement": True,  # required by PPO multi-gpu
    # },
    # "num_gpus": 2,
}

config.update(extra_config)

ray.init()
trainer = ppo.PPOTrainer(env=RaySeegaEnv, config=config)
policy = trainer.get_policy()
print(policy.model.base_model)

for i in count():
    results = trainer.train()
    print(pretty_print(results))

    if i% SAVE_STEP == 0:
        checkpoint = trainer.save()
        print(f"Checkpoint saved at {checkpoint}")


