import argparse
import numpy as np
from game import SnakeGame
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from qlearning_agent import QLearningAgent

def train(agent_type):
    game = SnakeGame(width=10, height=10)
    
    if agent_type == "dqn":
        state_size = len(game._get_state())
        action_size = 3
        agent = DQNAgent(state_size, action_size)
    elif agent_type == "qlearning":
        agent = QLearningAgent()
    else:
        raise ValueError("Invalid agent type. Choose 'dqn' or 'qlearning'")

    scores = []
    avg_scores = []
    
    episodes = 2000 if agent_type == "dqn" else 5000
    for episode in range(episodes):
        state = game.reset()
        done = False
        total_reward = 0
        
        while not done:
            if agent_type == "dqn":
                action = agent.act(state)
            else:
                action = agent.get_action(state)

            next_state, reward, done = game.step(action)
            total_reward += reward

            if agent_type == "dqn":
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
            else:
                agent.update_q_table(state, action, reward, next_state)

            state = next_state

            if episode % 100 == 0:
                game.render(speed=20)

        scores.append(game.score)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)
        
        if agent_type == "dqn":
            print(f"DQN Episode {episode+1}/{episodes} | Score: {game.score} | Avg: {avg_score:.2f} | ε: {agent.epsilon:.2f}")
        else:
            print(f"Q-Learning Episode {episode+1}/{episodes} | Score: {game.score} | Avg: {avg_score:.2f} | ε: {agent.epsilon:.2f}")

    plt.plot(avg_scores)
    plt.title(f"{agent_type.upper()} Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Average Score (100 episodes)")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL agents for Snake game')
    parser.add_argument('--agent', type=str, choices=['dqn', 'qlearning'], 
                        default='dqn', help='Type of agent to train')
    args = parser.parse_args()
    
    train(args.agent)