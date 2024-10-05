import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import argparse

from stable_baselines3.common.vec_env import DummyVecEnv

model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env, sb3_algo):
  match sb3_algo:
    case "SAC":
      model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    case "PPO":
      model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    case _:
      print("Algorithm not found")
      return

  TIMESTEPS = 250000

  model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
  model.save(f"{model_dir}/{sb3_algo}")

def test(env, sb3_algo, path_to_model):
    env = DummyVecEnv([lambda: env])
    match sb3_algo:
        case "SAC":
            model = SAC.load(path_to_model, env=env)
        case "PPO":
            model = PPO.load(path_to_model, env=env)
        case _:
            print("Algorithm not found")
            return

    obs = env.reset()
    done = False
    extra_steps = 500

    while extra_steps > 0:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)

        if done:
            extra_steps -= 1
            if extra_steps <= 0:
                break

    print("Testing complete")

def evaluate(env, sb3_algo, path_to_model, n_eval_episodes=100):
    match sb3_algo:
        case "SAC":
            model = SAC.load(path_to_model, env=env)
        case "PPO":
            model = PPO.load(path_to_model, env=env)
        case _:
            print("Algorithm not found")
            return

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Mean reward over {n_eval_episodes} episodes: {mean_reward:.2f}")
    print(f"Standard deviation of reward: {std_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, test, or evaluate model.")
    parser.add_argument("gymenv", help= "Gym environment Pendulum-v1")
    parser.add_argument("sb3_algo", help= "Stable Baselines RL algorithm PPO or SAC")
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-s", "--test", metavar="path_to_model")
    parser.add_argument("-e", "--evaluate", metavar="path_to_model")
    args = parser.parse_args()

    if args.train:
      env = gym.make(args.gymenv, render_mode=None)
      train(env, args.sb3_algo)

    # python pendulum.py Pendulum-v1 PPO -t

    if args.test:
     env = gym.make(args.gymenv, render_mode="human")
     test(env, args.sb3_algo, path_to_model=args.test)

    # python pendulum.py Pendulum-v1 PPO -s ./models/PPO_<timesteps>

    if args.evaluate:
     env = gym.make(args.gymenv, render_mode=None)
     evaluate(env, args.sb3_algo, path_to_model=args.evaluate)

    # python pendulum.py Pendulum-v1 PPO -e ./models/PPO

    else:
     print("No action taken")

  # tensorboard --logdir logs