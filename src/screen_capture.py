import os
import mss
import cv2
import datetime
import numpy as np
def init_screenshot(template_path=".\\assets\\templates\\e.png"):
    
    full_screen_bgr = capture_full_screen()
    region = detect_game_window(full_screen_bgr, template_path)
    if region == None:
        return region
    top_left, bottom_right = region
    print(f"Detected game window coordinates: {top_left} -> {bottom_right}")
    game_window_region = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
    
    return game_window_region
  
  
def capture_full_screen():
    with mss.mss() as sct:
        full_screen = sct.grab(sct.monitors[1])  
        full_screen_np = np.array(full_screen) 
        full_screen_bgr = cv2.cvtColor(full_screen_np, cv2.COLOR_RGBA2RGB)
        return full_screen_bgr

def detect_game_window(full_screen, template_path):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("The template image was not found!")
        return None
    full_screen_gray = cv2.cvtColor(full_screen, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(full_screen_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    threshold = 0.7
    if max_val < threshold:
        print("No matching game window found!")
        return None
    template_height, template_width = template.shape
    top_left = max_loc
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
    return top_left, bottom_right

def capture_game_window(region,count):
    
    count = count - 3
    save_location=".\\history"
    os.makedirs(save_location, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%d%H%M%S%f")[:-3]
        
    with mss.mss() as sct:
        left, top, right, bottom = region
        width, height = right - left, bottom - top
        monitor = {"top": top, "left": left, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        game_window_np = np.array(screenshot)
        game_window_bgr = cv2.cvtColor(game_window_np, cv2.COLOR_RGBA2RGB)
    if count > 0:
        save_path = os.path.join(save_location, f"{count}_complete.png")
        cv2.imwrite(save_path, game_window_bgr)
    return game_window_bgr