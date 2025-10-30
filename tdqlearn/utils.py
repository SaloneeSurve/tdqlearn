import torch
import matplotlib.pyplot as plt

def save_model(agent, path="dqn_model.pth"):
    torch.save(agent.policy_net.state_dict(), path)

def load_model(agent, path="dqn_model.pth"):
    agent.policy_net.load_state_dict(torch.load(path))
    agent.update_target()

def plot_rewards(rewards):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()
