# Doom AI Player with Deep Q-Learning (DQN) 🎮🤖

An AI agent trained using **Reinforcement Learning (DQN)** to play *Doom* through the VizDoom environment. The agent learns to navigate maps and shoot enemies autonomously.


## Key Features
- **Deep Q-Network (DQN)** implementation from scratch
- **Frame preprocessing** (grayscale, resizing)
- **Experience replay** for stable training
- **Action visualization** (OpenCV overlay)
- Customizable reward system

## Requirements
- Python 3.8+
- `vizdoom` (`pip install vizdoom`)
- `pytorch` (`pip install torch`)
- `opencv` (`pip install opencv-python`)

## Usage
1. Download `basic.wad` and place in project root
2. Run training:
   ```bash
   python doom_bot.py
