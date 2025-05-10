from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat, Button
import numpy as np
import time
import cv2

ENEMY_COLOR_LOWER = np.array([150, 0, 0])  
ENEMY_COLOR_UPPER = np.array([255, 70, 70])

def initialize_game():
    game = DoomGame()
    game.load_config("basic.cfg")  
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.RGB24)

    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.LOOK_UP)    
    game.add_available_button(Button.LOOK_DOWN)  
    game.add_available_button(Button.ATTACK)

    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)  

    game.init()
    return game

def detect_enemies(screen):

    frame = np.transpose(screen, (1, 2, 0))
    

    mask = cv2.inRange(frame, ENEMY_COLOR_LOWER, ENEMY_COLOR_UPPER)
    

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 50:  
        return None
        
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None

def enemy_targeting_logic(game):
    print("Starting enemy targeting test...")

    while not game.is_episode_finished():
        state = game.get_state()
        if state is None:
            continue
            
        screen = state.screen_buffer

        enemy_position = detect_enemies(screen)
        

        action = [0, 0, 0, 0, 0]
        
        if enemy_position:
            # Get frame dimensions
            h, w = screen.shape[1], screen.shape[2]
            center_x, center_y = w // 2, h // 2
            
            dx = enemy_position[0] - center_x
            dy = enemy_position[1] - center_y
            
            # Horizontal movement
            if dx < -10:  # Target is to the left
                action[0] = 1  # Turn left
                print("Turning Left")
            elif dx > 10:  # Target is to the right
                action[1] = 1  # Turn right
                print("Turning Right")
                
            # Vertical movement
            if dy < -10:  # Target is above
                action[2] = 1  # Look up
                print("Looking Up")
            elif dy > 10:  # Target is below
                action[3] = 1  # Look down
                print("Looking Down")
            
            # If target is centered, shoot
            if abs(dx) <= 15 and abs(dy) <= 15:
                action[4] = 1  # Attack
                print("FIRE!")
        else:
            # If no enemy detected, scan the environment
            frame = game.get_episode_time()
            if (frame // 20) % 2 == 0:
                action[0] = 1  # Turn left
                print("Scanning Left")
            else:
                action[1] = 1  # Turn right
                print("Scanning Right")
        

        frame = np.transpose(screen, (1, 2, 0))
        
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2-10, h//2), (w//2+10, h//2), (0, 255, 0), 2)
        cv2.line(frame, (w//2, h//2-10), (w//2, h//2+10), (0, 255, 0), 2)
        

        if enemy_position:
            cv2.circle(frame, enemy_position, 10, (0, 0, 255), 2)
            cv2.line(frame, (w//2, h//2), enemy_position, (255, 0, 0), 1)
  
        action_text = f"Action: {get_action_label(action)}"
        cv2.putText(frame, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("DOOM AI Targeting", frame)
        if cv2.waitKey(28) == 27:  # ESC to exit
            break

        game.make_action(action)
        time.sleep(0.028)  # ~35 fps

    print("Episode finished.")
    game.close()
    cv2.destroyAllWindows()

def get_action_label(action):
    keys = []
    if action[0]: keys.append("Left")
    if action[1]: keys.append("Right")
    if action[2]: keys.append("Up")
    if action[3]: keys.append("Down")
    if action[4]: keys.append("Fire")
    return " + ".join(keys) if keys else "NONE"

if __name__ == "__main__":
    game = initialize_game()
    game.new_episode()
    enemy_targeting_logic(game)