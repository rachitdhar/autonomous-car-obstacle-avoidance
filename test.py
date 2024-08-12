import gym
from numpy import random
from stable_baselines3 import PPO
from CarNavigationGymEnv import MODEL_REGISTER_NAME, CarNavigationEnv, register_env


def evaluate_random_policy(env, num_episodes=10, toRender = True):
    avg_reward = 0.0
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, done, info = env.step(action)
            if toRender: env.render(mode='human')
            total_reward += reward
        
        avg_reward += total_reward
        print(f"Random Policy - Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    
    avg_reward /= num_episodes
    print(f"\nRandom Policy - Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward


def test_trained_model(model_path, num_episodes=10, toRender = True):
    # Register the environment
    register_env()

    # Create the environment
    env = gym.make(MODEL_REGISTER_NAME)

    # Load the trained model
    model = PPO.load(model_path)

    # Testing random policy
    evaluate_random_policy(env, num_episodes, toRender)
    print()

    # Testing the trained agent
    avg_reward = 0.0
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if toRender: env.render(mode='human')
            total_reward += reward
            
        avg_reward += total_reward
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    
    avg_reward /= num_episodes
    print(f"\nTrained model - Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    env.close()

if __name__ == '__main__':
    test_trained_model("ppo_car_navigation", num_episodes=1000, toRender = False)
