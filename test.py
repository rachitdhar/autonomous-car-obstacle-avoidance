import gym
from stable_baselines3 import PPO
from CarNavigationGymEnv import MODEL_REGISTER_NAME, CarNavigationEnv, register_env

def test_trained_model(model_path, num_episodes=10):
    # Register the environment
    register_env()

    # Create the environment
    env = gym.make(MODEL_REGISTER_NAME)

    # Load the trained model
    model = PPO.load(model_path)

    # Testing the trained agent
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render(mode='human')
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    
    env.close()

if __name__ == '__main__':
    test_trained_model("ppo_car_navigation", num_episodes=100)
