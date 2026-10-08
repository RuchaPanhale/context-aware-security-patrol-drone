from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rl_navigation_env import DroneNavigationEnv


def main():
    env = DroneNavigationEnv()
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        gamma=0.98,
    )

    print("Training PPO navigation agent...")
    model.learn(total_timesteps=80000)
    model.save("ppo_drone_navigation")

    print("Training complete.")
    print("Saved model: ppo_drone_navigation.zip")


if __name__ == "__main__":
    main()
