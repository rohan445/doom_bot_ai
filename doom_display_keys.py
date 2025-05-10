import cv2 
from vizdoom import DoomGame
import numpy as np 
import time 

def initialize_doom():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_window_visible(True)  # Changed to True to display the game window
    game.set_mode(0)
    game.init()
    return game 

actions = [
    [1, 0, 0],  
    [0, 1, 0],  
    [0, 0, 1],  
    [1, 0, 1],  #to move left and right 
]

def get_action_label(action):
    keys = []
    if action[0]: keys.append("W")
    if action[1]: keys.append("A")
    if action[2]: keys.append("D")
    return "+".join(keys)

game = initialize_doom()

for episode in range(5):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        screen_buf = state.screen_buffer
        frame = np.transpose(screen_buf, (1, 2, 0))  # Fixed shape transformation
        action = actions[np.random.randint(len(actions))]
        reward = game.make_action(action)
        
        label = get_action_label(action)
        cv2.putText(frame, f"AI input: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("DOOM AI", frame)
        if cv2.waitKey(20) == 27:  
            break
        
game.close()
cv2.destroyAllWindows()