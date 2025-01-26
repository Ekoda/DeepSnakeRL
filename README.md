# Snake Reinforcement Learning Experiment with DeepSeek-R1

This project explores reinforcement learning (RL) techniques for the classic Snake game, developed while experimenting with DeepSeek's new R1 large language model. The implementation compares Q-Learning and Deep Q-Network (DQN) approaches while leveraging DeepSeek-R1's code generation and problem-solving capabilities.

## 🚀 Overview

A Python implementation of Snake with two RL agent options:
- **Q-Learning** with discrete state representation
- **Deep Q-Network (DQN)** with continuous state space
- Integrated Pygame visualization
- Training metrics tracking

Developed in collaboration with DeepSeek-R1 to explore:
1. LLM-assisted code generation
2. RL algorithm implementation
3. Performance optimization strategies
4. State representation design

## ✨ Features

- 🐍 Customizable Snake game environment
- 🤖 Dual agent architecture (Q-Learning + DQN)
- 🧠 Enhanced state representations:
  - Basic (11 binary features)
  - Advanced (21 continuous features)
- 📊 Real-time visualization and training metrics
- 🔄 CLI interface for agent selection

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/snake-rl-deepseek.git
cd snake-rl-deepseek

# Install dependencies
pip install -r requirements.txt

# Train DQN agent (default)
python main.py --agent dqn

# Train Q-Learning agent
python main.py --agent qlearning