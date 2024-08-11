import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from CarNavigationGymEnv import MODEL_REGISTER_NAME
from CarNavigationGymEnv import CarNavigationEnv, register_env


LEARNING_RATE = 3e-4
N_STEPS = 4096
BATCH_SIZE = 256
TOTAL_TIMESTEPS = 2_000_000

def main(test = True):
    # Register the environment
    register_env()

    # Create the environment
    env = gym.make(MODEL_REGISTER_NAME)

    # Initialize the agent
    model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE)

    # Train the agent
    total_timesteps = TOTAL_TIMESTEPS
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save the trained model
    model.save("ppo_car_navigation")

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    if test:
        # Test the trained agent
        obs = env.reset()
        for _ in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render(mode='human')
            if done:
                obs = env.reset()
    
    # plot paths taken during training episodes
    env.display_all_paths(savePlotData = True)
    env.close()


if __name__ == '__main__':
    main(test = False)
