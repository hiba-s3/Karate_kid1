import cv2
import numpy as np
import mss

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

def capture_game_window(region):
    with mss.mss() as sct:
        left, top, right, bottom = region
        width, height = right - left, bottom - top
        monitor = {"top": top, "left": left, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        game_window_np = np.array(screenshot)
        game_window_bgr = cv2.cvtColor(game_window_np, cv2.COLOR_RGBA2RGB)
        return game_window_bgr

template_path = "C:/Users/asus/CV_final_project/assets/e.png"

full_screen_bgr = capture_full_screen()

region = detect_game_window(full_screen_bgr, template_path)
if region is None:
    print("Failed to detect the game window.")
else:
    top_left, bottom_right = region
    print(f"Detected game window coordinates: {top_left} -> {bottom_right}")
    
    game_window_region = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
    
    while True:  
        game_window = capture_game_window(game_window_region)
        
        cv2.imshow("Game Window", game_window)
        cv2.imwrite("game_window.png", game_window)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
