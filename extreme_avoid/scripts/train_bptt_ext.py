#!/usr/bin/env python3
"""
Extended BPTT Training Script.

COPY of depthnav/scripts/train_bptt.py with two critical changes (per skill.md §0.1, §2.3):

1. Import env/policy aliases from extreme_avoid.registry (extended dictionaries)
   instead of depthnav's original alias files.
2. Replace hardcoded `if policy_class == MultiInputPolicy:` with
   `if issubclass(policy_class, MultiInputPolicy):` to support subclasses
   like TrackerFusedPolicy.

3. (NEW) Pass aux_dict from policy to env for Innovation 2 motion prediction.

All other logic (yaml loading, BPTT construction, learn/deploy) is preserved.
"""

import faulthandler
faulthandler.enable()

import sys
import os
import yaml
import torch as th
import argparse
from copy import deepcopy

# --- CRITICAL: Import from extended registry (not depthnav originals) ---
# Make sure code/ is on PYTHONPATH so extreme_avoid is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from extreme_avoid.registry import env_aliases, policy_aliases
from extreme_avoid.scripts.eval_avoidance import EvaluateDynamicAvoidance
from extreme_avoid.vendor.depthnav.policies.bptt_algorithm import BPTT
from extreme_avoid.vendor.depthnav.policies.mlp_policy import MlpPolicy
from extreme_avoid.vendor.depthnav.policies.multi_input_policy import MultiInputPolicy
from extreme_avoid.vendor.depthnav.common import ExitCode


def convert_observations_to_device(obs, device):
    """Move observation dict tensors to target device."""
    import numpy as np
    if isinstance(obs, dict):
        return {k: th.as_tensor(v, device=device) if isinstance(v, (np.ndarray, th.Tensor)) else v
                for k, v in obs.items()}
    return obs


def main(args):
    # --- Load training env ---
    with open(args.cfg_file, "r") as file:
        config = yaml.safe_load(file)

    env_class = env_aliases[config["env_class"]]
    env = env_class(requires_grad=True, **config["env"])

    # --- Load eval envs ---
    eval_envs = []
    if args.eval_configs is not None:
        for cfg_file in args.eval_configs:
            with open(cfg_file, "r") as file:
                eval_config = yaml.safe_load(file)

            if args.render:
                eval_config["scene_kwargs"]["render_settings"] = {
                    "mode": "follow",
                    "view": "back",
                    "sensor_type": "color",
                    "resolution": [512, 512],
                    "axes": True,
                    "trajectory": False,
                    "object_path": "./datasets/depthnav_dataset/configs/agents/DJI_Mavic_Mini_2.object_config.json",
                    "line_width": 2.0,
                }

            eval_env_class = env_aliases[config["env_class"]]
            eval_env = eval_env_class(requires_grad=False, **eval_config["env"])
            eval_envs.append(eval_env)

    # --- Load policy ---
    policy_class = policy_aliases[config["policy_class"]]
    policy_kwargs = config["policy"]

    # FIXED: Use issubclass to support TrackerFusedPolicy (MultiInputPolicy subclass)
    if issubclass(policy_class, MultiInputPolicy):
        policy = policy_class(env.observation_space, **policy_kwargs)
    else:
        policy = policy_class(**policy_kwargs)

    if args.weight is not None:
        policy.load(args.weight)
    elif config.get("weights_file", None):
        policy.load(config["weights_file"])

    # --- Setup trainer ---
    trainer = BPTT(
        env=env,
        eval_envs=eval_envs,
        eval_csvs=args.eval_csvs,
        policy=policy,
        run_name=args.run_name,
        logging_dir=args.logging_root,
        **config["train_bptt"],
    )

    # --- Train (with extended policy env integration for aux_dict) ---
    print("Starting BPTT training with extended policy...")
    trainer.policy.train()

    from tqdm import tqdm

    exit_code = ExitCode.ERROR
    start_iter = args.start_iter

    try:
        env.reset()
        episode_steps = 0
        latent_state = th.zeros(
            (env.num_envs, trainer.policy.latent_dim) if trainer.policy.is_recurrent else (env.num_envs, 1),
            device=trainer.policy.device,
        )

        for iter in tqdm(range(trainer.iterations)):
            trainer.policy.train()
            loss = 0.0
            discount_factor = th.ones(env.num_envs, dtype=th.float32, device=trainer.device)

            if episode_steps >= env.max_episode_steps:
                env.reset()
                episode_steps = 0

            for _ in range(trainer.horizon):
                obs = env.get_observation()
                
                # Check if policy is bare MlpPolicy (no feature extractor, takes state tensor directly)
                if type(trainer.policy) is MlpPolicy:
                    obs_device = convert_observations_to_device(obs, trainer.policy.device)
                    actions = trainer.policy(obs_device["state"])
                    aux_dict = {}
                # Check if policy supports extended forward (TrackerFusedPolicy)
                elif hasattr(trainer.policy, 'motion_head') and trainer.policy._has_motion_head:
                    obs_device = convert_observations_to_device(obs, trainer.policy.device)
                    if trainer.policy.is_recurrent:
                        actions, aux_dict, latent_state = trainer.policy(obs_device, latent_state, return_aux=True)
                    else:
                        actions = trainer.policy(obs_device)
                        aux_dict = {}
                else:
                    obs_device = convert_observations_to_device(obs, trainer.policy.device)
                    if trainer.policy.is_recurrent:
                        actions, latent_state = trainer.policy(obs_device, latent_state)
                        aux_dict = {}
                    else:
                        actions = trainer.policy(obs_device)
                        aux_dict = {}

                # Pass aux_dict to env for reward computation (Innovation 2+3)
                if hasattr(env, 'set_obstacle_prediction'):
                    env.set_obstacle_prediction(aux_dict)

                # Step
                obs, reward, done, info = env.step(actions, is_test=False)
                reward = reward.to(trainer.device)
                done = done.to(trainer.device).to(th.bool)
                loss = loss + -1.0 * reward * discount_factor

                discount_factor = discount_factor * trainer.gamma * ~done + done
                latent_state = latent_state * ~done.unsqueeze(1)

            episode_steps += trainer.horizon

            loss = loss / trainer.horizon
            loss = loss.mean()
            trainer.optimizer.zero_grad()
            loss.backward()

            max_norm = 5.0
            grad_norm = th.nn.utils.clip_grad_norm_(
                trainer.policy.parameters(), max_norm=max_norm
            )
            print(f"grad norm = {grad_norm:.4f}")

            trainer.optimizer.step()
            trainer.lr_schedule.step()

            env.detach()
            latent_state = latent_state.clone().detach()

            # Logging
            if iter % trainer.log_interval == 0:
                trainer.policy.eval()
                for i, (eval_env, csv_file) in enumerate(
                    zip(trainer.eval_envs, trainer.eval_csvs)
                ):
                    e = EvaluateDynamicAvoidance(eval_env, trainer.policy)
                    index = start_iter + iter
                    df = e.run_rollouts(num_rollouts=5, run_name=index, render=args.render)
                    df["scene"] = os.path.basename(env.scene_manager.scene_path)
                    basename = os.path.basename(csv_file).split(".")[0]
                    trainer.df_to_tensorboard(trainer._logger, df, prefix=basename)
                    write_header = not os.path.exists(csv_file)
                    df.to_csv(csv_file, float_format="%.3f", mode="a",
                              header=write_header, columns=trainer.whitelisted_csv_keys)
                    print(f"wrote stats to {csv_file}")

                trainer._logger.record("train/learning_rate", trainer.lr_schedule.get_last_lr()[0])
                trainer._logger.record("train/loss", float(loss))
                trainer._logger.record("train/grad_norm", float(grad_norm))
                trainer._logger.dump(start_iter + iter)

            if iter > 0 and iter % trainer.checkpoint_interval == 0:
                trainer.save(trainer.run_path + "_iteration_" + str(iter) + ".pth")

        exit_code = ExitCode.SUCCESS
    except KeyboardInterrupt:
        trainer.save(trainer.run_path + "_iteration_" + str(iter) + ".pth")
        exit_code = ExitCode.KEYBOARD_INTERRUPT
    finally:
        for eval_env in trainer.eval_envs:
            eval_env.close()
        env.close()

    print("Done training. Saving model")
    trainer.save()
    sys.exit(exit_code.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="extreme_avoid/configs/train_cfg/stage3_perception.yaml")
    parser.add_argument("--logging_root", type=str, default="./logs")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--eval_configs", nargs="+", type=str, default=None)
    parser.add_argument("--eval_csvs", nargs="+", type=str, default=None)
    args = parser.parse_args()
    main(args)
