
import time
import keyboard  

def waitKey():
    """Wait for the user to press 'e' globally to start the game capture."""
    print("Press 'e' to start the game capture...")
    while True:
        if keyboard.is_pressed('e'):  
            break
    print("Starting game capture...")
import keyboard
import time
def trigger_event(position):
    """Simulate pressing the right or left arrow key."""
     
    
    if position == "right":
        keyboard.press('right')         
        time.sleep(0.01)
        keyboard.release('right') 
    elif position == "left":
        keyboard.press('left')        
        time.sleep(0.01)
        keyboard.release('left')
    else:
        print("error")