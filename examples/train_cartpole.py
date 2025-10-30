# examples/train_cartpole.py
import gym
import torch
from tdqlearn import DQNAgent, plot_rewards

env = gym.make("CartPole-v1")
agent = DQNAgent(state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.n)

num_episodes = 200
target_update = 10
rewards = []

for episode in range(num_episodes):
    state = env.reset()[0]
    total_reward = 0

    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.memory.push(state, action, reward, next_state, done)
        agent.optimize()
        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    if episode % target_update == 0:
        agent.update_target()

    print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward}")

plot_rewards(rewards)
