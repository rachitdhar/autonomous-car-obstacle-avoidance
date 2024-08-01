import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from CarNavigationGymEnv import MODEL_REGISTER_NAME
from CarNavigationGymEnv import CarNavigationEnv, register_env

def main():
    # Register the environment
    register_env()

    # Create the environment
    env = gym.make(MODEL_REGISTER_NAME)
    #env = DummyVecEnv([lambda: gym.make(MODEL_REGISTER_NAME)])  # Wrap in a DummyVecEnv

    # Initialize the agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64)

    # Train the agent
    total_timesteps = 10000  # Adjust this based on your needs
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save the trained model
    model.save("ppo_car_navigation")

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Test the trained agent
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(mode='human')
        if done:
            obs = env.reset()
    
    env.close()


if __name__ == '__main__':
    main()
