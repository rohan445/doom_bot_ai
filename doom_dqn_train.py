import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import cv2
from vizdoom import *
import os
from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat, Button
from dqn_model import DQN
from memory import ReplayMemory, Transition

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Linear(conv_out_size, 512)
        self.out = nn.Linear(512, n_actions)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv3(self.conv2(self.conv1(o)))
        return int(o.view(1, -1).size(1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.out(x)

def preprocess(img):
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (84, 84))
    return img / 255.0

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)
    
    def reset(self):
        self.frames.clear()
    
    def push(self, frame):
        self.frames.append(frame)
    
    def get_state(self):
        if len(self.frames) < self.k:
            padding = [self.frames[-1]] * (self.k - len(self.frames)) if self.frames else [np.zeros((84, 84), dtype=np.float32)] * self.k
            return np.stack(list(padding) + list(self.frames), axis=0)
        else:
            return np.stack(self.frames, axis=0)

def setup_game():
    game = DoomGame()
    
    possible_paths = [
        "basic.cfg",
        "scenarios/basic.cfg",
        "vizdoom/scenarios/basic.cfg",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "basic.cfg"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios/basic.cfg")
    ]
    
    cfg_found = False
    for cfg_path in possible_paths:
        if os.path.exists(cfg_path):
            game.load_config(cfg_path)
            cfg_found = True
            print(f"Found config at: {cfg_path}")
            break
    
    if not cfg_found:
        print("Config file not found. Creating minimal configuration.")
        game.set_doom_scenario_path("basic.wad")
        game.set_doom_map("map01")
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_living_reward(-1)
    
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    
    game.add_available_button(Button.MOVE_FORWARD)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.ATTACK)
    
    if hasattr(Button, 'LOOK_UP'):
        game.add_available_button(Button.LOOK_UP)
    if hasattr(Button, 'LOOK_DOWN'):
        game.add_available_button(Button.LOOK_DOWN)
    
    try:
        game.set_labels_buffer_enabled(True)
    except:
        print("Labels buffer not supported in this version. Enemy detection will be disabled.")
    
    game.set_window_visible(True)
    
    try:
        game.init()
        print("Game initialized successfully!")
        return game
    except Exception as e:
        print(f"Failed to initialize game: {e}")
        raise

def detect_enemies(labels):
    for l in labels:
        if l.object_name == "DoomPlayer":
            return True, l.x, l.y
    return False, 0, 0

def calculate_aim_adjustment(enemy_x, enemy_y):
    center_x = 320
    center_y = 240
    dx = enemy_x - center_x
    dy = enemy_y - center_y
    
    if abs(dx) > 30 or abs(dy) > 30:
        if abs(dx) > abs(dy):
            if dx < 0:
                return 6
            else:
                return 7
        else:
            if dy < 0:
                return 9
            else:
                return 10
    return 8

def save_model(model, path="models/doom_dqn.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

buffer_size = 100000
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.0001
target_update_freq = 10
num_episodes = 1000
stack_size = 4
eval_frequency = 20

try:
    game = setup_game()
except Exception as e:
    print(f"Critical error setting up game: {e}")
    print("Please check your VizDoom installation and configuration files.")
    import sys
    sys.exit(1)

available_buttons = game.get_available_buttons_size()
print(f"Available buttons: {available_buttons}")

if available_buttons == 4:
    n_actions = 8
    actions = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
    ]
elif available_buttons == 6:
    n_actions = 11
    actions = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1]
    ]
else:
    n_actions = 2**available_buttons
    actions = []
    for i in range(n_actions):
        action = [0] * available_buttons
        for j in range(available_buttons):
            if (i >> j) & 1:
                action[j] = 1
        actions.append(action)
    print(f"Generated {n_actions} actions for {available_buttons} buttons")
    if n_actions > 20:
        n_actions = 20
        actions = actions[:20]
        print(f"Limited to {n_actions} actions")

model = DQN((stack_size, 84, 84), n_actions).to(device)
target_model = DQN((stack_size, 84, 84), n_actions).to(device)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
buffer = ReplayBuffer(buffer_size)
frame_stack = FrameStack(stack_size)

rewards_tensor = torch.zeros(batch_size, device=device)
dones_tensor = torch.zeros(batch_size, device=device)

best_eval_reward = float('-inf')

for episode in range(num_episodes):
    game.new_episode()
    frame_stack.reset()
    
    first_frame = preprocess(game.get_state().screen_buffer)
    for _ in range(stack_size):
        frame_stack.push(first_frame)
    
    state = frame_stack.get_state()
    total_reward = 0
    steps = 0
    
    while not game.is_episode_finished():
        state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32).to(device)
        
        if random.random() < epsilon:
            action_idx = random.randrange(n_actions)
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                action_idx = torch.argmax(q_values).item()
        
        if not game.is_episode_finished():
            try:
                labels = game.get_state().labels
                enemy_visible, enemy_x, enemy_y = detect_enemies(labels)
                if enemy_visible and random.random() < 0.9:
                    action_idx = calculate_aim_adjustment(enemy_x, enemy_y)
            except:
                pass
        
        reward = game.make_action(actions[action_idx])
        done = game.is_episode_finished()
        
        if not done:
            next_frame = preprocess(game.get_state().screen_buffer)
            frame_stack.push(next_frame)
            next_state = frame_stack.get_state()
        else:
            next_state = np.zeros_like(state)
        
        buffer.push(state, action_idx, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if len(buffer) > batch_size:
            states, actions_b, rewards, next_states, dones = buffer.sample(batch_size)
            
            states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            actions_tensor = torch.tensor(actions_b, dtype=torch.long).to(device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
            
            q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_q_values = target_model(next_states_tensor).max(1)[0]
                expected_q = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
            
            loss = F.mse_loss(q_values, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if episode % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {epsilon:.4f}")
    
    if episode % eval_frequency == 0 or episode == num_episodes - 1:
        eval_rewards = []
        for _ in range(5):
            game.new_episode()
            frame_stack.reset()
            
            first_frame = preprocess(game.get_state().screen_buffer)
            for _ in range(stack_size):
                frame_stack.push(first_frame)
            
            eval_state = frame_stack.get_state()
            eval_total_reward = 0
            
            while not game.is_episode_finished():
                eval_state_tensor = torch.tensor(np.expand_dims(eval_state, axis=0), dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    q_values = model(eval_state_tensor)
                    action_idx = torch.argmax(q_values).item()
                
                if not game.is_episode_finished():
                    try:
                        labels = game.get_state().labels
                        enemy_visible, enemy_x, enemy_y = detect_enemies(labels)
                        if enemy_visible:
                            action_idx = calculate_aim_adjustment(enemy_x, enemy_y)
                    except:
                        pass
                
                eval_reward = game.make_action(actions[action_idx])
                eval_done = game.is_episode_finished()
                
                if not eval_done:
                    eval_next_frame = preprocess(game.get_state().screen_buffer)
                    frame_stack.push(eval_next_frame)
                    eval_state = frame_stack.get_state()
                
                eval_total_reward += eval_reward
            
            eval_rewards.append(eval_total_reward)
        
        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
        print(f"Evaluation - Average Reward: {avg_eval_reward:.2f}")
        
        if episode == 0 or avg_eval_reward > best_eval_reward:
            best_eval_reward = avg_eval_reward
            save_model(model, f"models/doom_dqn_best.pth")
    
    if episode % 100 == 0:
        save_model(model, f"models/doom_dqn_episode_{episode}.pth")

save_model(model, "models/doom_dqn_final.pth")
print("Training completed!")