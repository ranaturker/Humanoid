import gymnasium as gym
from stable_baselines3 import SAC, PPO, DDPG
import os
import argparse

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
    case "DDPG":
      model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    case _:
      print("Algorithm not found")
      return

  TIMESTEPS = 2500

  model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
  model.save(f"{model_dir}/{sb3_algo}")


def test(env, sb3_algo, path_to_model):
    match sb3_algo:
        case "SAC":
            model = SAC.load(path_to_model, env=env)
        case "PPO":
            model = PPO.load(path_to_model, env=env)
        case "DDPG":
            model = DDPG.load(path_to_model, env=env)
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train or test model.")
  parser.add_argument("gymenv", help= "Gym environment Humanoid-v4")
  parser.add_argument("sb3_algo", help= "Stable Baselines RL algorithm PPO, SAC or DDPG")
  parser.add_argument("-t", "--train", action="store_true")
  parser.add_argument("-s", "--test", metavar="path_to_model")
  args = parser.parse_args()

  if args.train:
    env = gym.make(args.gymenv, render_mode=None)
    train(env, args.sb3_algo)

      #python humanoid.py Humanoid-v4 PPO --train

  if args.test:
    env = gym.make(args.gymenv, render_mode="human")
    test(env, args.sb3_algo, path_to_model= args.test)

    # python humanoid.py Humanoid-v4 PPO -t
    # tensorboard --logdir logs
    # python humanoid.py Humanoid-v4 PPO -s ./models/PPO_<timesteps>

  else:
    print("No action taken")