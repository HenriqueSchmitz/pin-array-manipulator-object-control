import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from pin_array_manipulator_object_control.environment.composite_control_env import CompositeControlEnv
from pin_array_manipulator_object_control.manipulator.observation import PinArrayEnvObservation
from pin_array_manipulator_object_control.manipulator.pin_array_manipulator import PinArrayManipulatorConfig
from pin_array_manipulator_object_control.objects.ball import Ball
from pin_array_manipulator_object_control.rewards.distance_3d import Distance3DRewardModel
from pin_array_manipulator_object_control.routines.multi_target_generator import MultiTargetGenerator


class SubgoalPoseRLEnv(CompositeControlEnv):
    def __init__(self, *args, manipulator_config: PinArrayManipulatorConfig, **kwargs):
        self.wrapper_max_episode_steps = int(kwargs.get("max_episode_steps", 1000))

        super().__init__(*args, manipulator_config=manipulator_config, **kwargs)

        self.config = manipulator_config

        raw_obs_dim = int(np.prod(self.observation_space.shape))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(raw_obs_dim + 3,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.delta_scale = np.array(
            [0.04, 0.04, 0.002, 0.05, 0.05, 0.05],
            dtype=np.float32,
        )

        self.base_seek_speed = 5e-4
        self.min_seek_speed = 1e-4
        self.control_hold_steps = 5

        self.success_radius = 0.03
        self.failure_radius = 0.60
        self.max_z_error = 0.25

        self.progress_weight = 300.0
        self.distance_weight = 0.5
        self.subgoal_alignment_weight = 0.0
        self.subgoal_forward_weight = 2.0
        self.time_penalty = 0.01
        self.success_reward = 100.0
        self.failure_penalty = 25.0

        self.step_count = 0
        self.current_obs = None

        self.episode_target = None
        self.base_object_z = None
        self.prev_object_pos = None
        self.prev_translation_error = None
        self.best_translation_error = None

        self.held_composite_action = None
        self.held_subgoal_pose = None
        self.held_pose_delta = None
        self.held_policy_action = None

    def _sample_nontrivial_target(self, object_pos):
        target = np.zeros(6, dtype=np.float32)
        target[:3] = object_pos[:3]
        target[3:] = 0.0

        min_dist = 0.10
        max_dist = 0.25

        angle = np.random.uniform(0.0, 2.0 * np.pi)
        radius = np.random.uniform(min_dist, max_dist)

        target[0] = object_pos[0] + radius * np.cos(angle)
        target[1] = object_pos[1] + radius * np.sin(angle)
        target[2] = object_pos[2]

        return target

    def _augment_obs(self, raw_obs):
        obs_obj = self._parse_obs(raw_obs)
        object_pos = obs_obj.object_pose.array()[:3].astype(np.float32)

        rel_xy = self.episode_target[:2].astype(np.float32) - object_pos[:2]
        dist = np.array([np.linalg.norm(rel_xy)], dtype=np.float32)

        return np.concatenate(
            [
                np.asarray(raw_obs, dtype=np.float32).reshape(-1),
                rel_xy,
                dist,
            ]
        ).astype(np.float32)

    def reset(self, **kwargs):
        raw_obs, info = super().reset(**kwargs)
        self.current_obs = raw_obs

        obs_obj = self._parse_obs(raw_obs)
        object_pose = obs_obj.object_pose.array().astype(np.float32)
        object_pos = object_pose[:3]

        self.base_object_z = float(object_pos[2])
        self.episode_target = self._sample_nontrivial_target(object_pos)

        self.step_count = 0
        self.prev_object_pos = object_pos.copy()
        self.prev_translation_error = self._xy_distance(object_pos, self.episode_target[:3])
        self.best_translation_error = self.prev_translation_error

        self.held_composite_action = None
        self.held_subgoal_pose = None
        self.held_pose_delta = None
        self.held_policy_action = None

        info["target_xy"] = self.episode_target[:2].copy()
        info["initial_distance"] = self.prev_translation_error

        return self._augment_obs(raw_obs), info

    def step(self, action):
        self.step_count += 1

        policy_action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )

        (
            composite_action,
            subgoal_pose,
            pose_delta,
            used_policy_action,
            action_was_updated,
        ) = self._get_or_update_held_action(policy_action)

        raw_obs, _, base_terminated, base_truncated, info = super().step(composite_action)
        self.current_obs = raw_obs

        reward, reward_info = self._compute_reward(
            obs=raw_obs,
            policy_action=used_policy_action,
            pose_delta=pose_delta,
        )

        self._update_reward_state(reward_info)

        success = reward_info["success"]
        failure = reward_info["failure"]

        terminated = bool(success or failure)

        timeout = self.step_count >= self.wrapper_max_episode_steps
        truncated = bool((timeout or base_truncated) and not terminated)

        info.update(reward_info)
        info["TimeLimit.truncated"] = truncated and not terminated
        info["is_success"] = bool(success)
        info["subgoal_pose"] = subgoal_pose
        info["pose_delta"] = pose_delta
        info["action_was_updated"] = action_was_updated

        return self._augment_obs(raw_obs), reward, terminated, truncated, info

    def _get_or_update_held_action(self, policy_action):
        should_update = (
            self.held_composite_action is None
            or (self.step_count - 1) % self.control_hold_steps == 0
        )

        if not should_update:
            return (
                self.held_composite_action,
                self.held_subgoal_pose,
                self.held_pose_delta,
                self.held_policy_action,
                False,
            )

        obs_obj = self._parse_obs(self.current_obs)
        current_pose = obs_obj.object_pose.array().astype(np.float32)

        pose_delta = policy_action * self.delta_scale
        subgoal_pose = current_pose + pose_delta

        composite_action = np.concatenate(
            (
                np.array([self.base_seek_speed, self.min_seek_speed], dtype=np.float32),
                subgoal_pose.astype(np.float32),
            )
        ).astype(np.float32)

        self.held_composite_action = composite_action
        self.held_subgoal_pose = subgoal_pose.copy()
        self.held_pose_delta = pose_delta.copy()
        self.held_policy_action = policy_action.copy()

        return (
            self.held_composite_action,
            self.held_subgoal_pose,
            self.held_pose_delta,
            self.held_policy_action,
            True,
        )

    def _compute_reward(self, obs, policy_action, pose_delta):
        obs_obj = self._parse_obs(obs)

        object_pose = obs_obj.object_pose.array().astype(np.float32)
        object_pos = object_pose[:3].copy()

        target_pos = self.episode_target[:3]
        prev_pos = self.prev_object_pos.copy()

        prev_dist = self._xy_distance(prev_pos, target_pos)
        curr_dist = self._xy_distance(object_pos, target_pos)
        progress = prev_dist - curr_dist

        move_vec = object_pos - prev_pos
        move_xy = move_vec[:2]
        move_xy_norm = float(np.linalg.norm(move_xy))

        goal_vec_xy = target_pos[:2] - prev_pos[:2]
        goal_norm_xy = float(np.linalg.norm(goal_vec_xy))

        if goal_norm_xy > 1e-8:
            goal_dir_xy = goal_vec_xy / goal_norm_xy
        else:
            goal_dir_xy = np.zeros(2, dtype=np.float32)

        if move_xy_norm > 1e-8 and goal_norm_xy > 1e-8:
            forward_motion = float(np.dot(move_xy, goal_dir_xy))
            alignment = float(forward_motion / (move_xy_norm + 1e-8))
        else:
            forward_motion = 0.0
            alignment = 0.0

        subgoal_xy = pose_delta[:2].astype(np.float32)
        subgoal_norm = float(np.linalg.norm(subgoal_xy))

        if subgoal_norm > 1e-8 and goal_norm_xy > 1e-8:
            subgoal_forward = float(np.dot(subgoal_xy, goal_dir_xy))
            subgoal_alignment = float(np.dot(subgoal_xy / subgoal_norm, goal_dir_xy))
        else:
            subgoal_forward = 0.0
            subgoal_alignment = 0.0

        z_error = abs(float(object_pos[2] - self.base_object_z))

        object_out_of_bounds = curr_dist > self.failure_radius
        object_fell = z_error > self.max_z_error

        success = curr_dist < self.success_radius
        failure = object_out_of_bounds or object_fell

        progress_reward = self.progress_weight * progress

        subgoal_alignment_reward = (
            self.subgoal_alignment_weight * max(subgoal_alignment, 0.0)
        )
        subgoal_forward_reward = (
            self.subgoal_forward_weight * max(subgoal_forward, 0.0)
        )

        subgoal_backward_penalty = -10.0 * max(-subgoal_forward, 0.0)

        distance_penalty = -self.distance_weight * curr_dist

        movement_bonus = 0.0
        if curr_dist > self.success_radius and move_xy_norm > 1e-5:
            movement_bonus = 0.03

        z_penalty = -0.25 * z_error
        action_penalty = -0.0005 * float(np.square(policy_action).sum())
        step_penalty = -self.time_penalty

        terminal_reward = 0.0
        if success:
            terminal_reward += self.success_reward
        if failure:
            terminal_reward -= self.failure_penalty

        reward_unclipped = (
            progress_reward
            + subgoal_alignment_reward
            + subgoal_forward_reward
            + subgoal_backward_penalty
            + distance_penalty
            + movement_bonus
            + z_penalty
            + action_penalty
            + step_penalty
            + terminal_reward
        )

        reward = float(np.clip(reward_unclipped, -25.0, 100.0))

        return reward, {
            "success": bool(success),
            "failure": bool(failure),
            "object_fell": bool(object_fell),
            "object_out_of_bounds": bool(object_out_of_bounds),

            "current_pose_error": curr_dist,
            "pose_error": curr_dist,
            "current_translation_error": curr_dist,
            "distance_to_target": curr_dist,
            "current_distance": curr_dist,

            "translation_progress": progress,
            "progress_xy": progress,

            "forward_motion": forward_motion,
            "alignment": alignment,

            "subgoal_forward": subgoal_forward,
            "subgoal_alignment": subgoal_alignment,

            "progress_reward": progress_reward,
            "subgoal_alignment_reward": subgoal_alignment_reward,
            "subgoal_forward_reward": subgoal_forward_reward,
            "subgoal_backward_penalty": subgoal_backward_penalty,
            "distance_penalty": distance_penalty,
            "movement_bonus": movement_bonus,
            "z_penalty": z_penalty,
            "action_penalty": action_penalty,
            "step_penalty": step_penalty,
            "terminal_reward": terminal_reward,

            "reward_unclipped": float(reward_unclipped),

            "move_norm": float(np.linalg.norm(move_vec)),
            "move_xy_norm": move_xy_norm,
            "z_movement": abs(float(move_vec[2])),
            "z_error": z_error,

            "current_object_pos": object_pos,
            "current_forward_sign": float(np.sign(progress)),
        }

    def _update_reward_state(self, reward_info):
        self.prev_translation_error = reward_info["current_translation_error"]
        self.best_translation_error = min(
            self.best_translation_error,
            self.prev_translation_error,
        )
        self.prev_object_pos = reward_info["current_object_pos"].copy()

    def _pose_error(self, obs):
        obs_obj = self._parse_obs(obs)
        object_pos = obs_obj.object_pose.array()[:3]
        return self._xy_distance(object_pos, self.episode_target[:3])

    def _translation_error(self, obs):
        return self._pose_error(obs)

    def _parse_obs(self, obs):
        return PinArrayEnvObservation.from_array(
            obs,
            self.config.pins_per_side,
        )

    @staticmethod
    def _xy_distance(a, b):
        return float(np.linalg.norm(np.asarray(a[:2]) - np.asarray(b[:2])))


def make_env():
    config = PinArrayManipulatorConfig(
        manipulator_size=1.0,
        pins_per_side=15,
        pin_height=0.15,
        actuation_length=0.1,
        pin_spacing=0.001,
        has_wall=True,
        rounded_pins=True,
    )

    ball = Ball(diameter=0.1, starting_z=0.2)

    reward_model = Distance3DRewardModel(manipulator_config=config)

    target_generator = MultiTargetGenerator(
        simulation_object=ball,
        manipulator_config=config,
    )

    env = SubgoalPoseRLEnv(
        simulation_object=ball,
        target_generator=target_generator,
        reward_model=reward_model,
        manipulator_config=config,
        render_mode=None,
        max_episode_steps=5000,
    )

    return Monitor(env)


def main():
    env = make_env()

    print("obs space:", env.observation_space)
    print("action space:", env.action_space)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        ),
        log_std_init=-0.2,
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="ppo_subgoal_logs/",
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0002,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.05,
        device="cpu",
    )

    model.learn(total_timesteps=50_000)
    model.save("ppo_subgoal_pin_array")

    env.close()


if __name__ == "__main__":
    main()