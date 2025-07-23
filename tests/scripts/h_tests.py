
import cv2 as cv
def find_object_in_screenshot(screenshot_path, template_path):
    
    screenshot = cv.imread(screenshot_path)
    template = cv.imread(template_path)
    if screenshot is None or template is None:
        print("Error loading images.")
        return
    
    result = cv.matchTemplate(screenshot, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    
    threshold = 0.8  
    if max_val >= threshold:
        print(f"Object found with confidence {max_val:.2f}")
        
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        
        cv.rectangle(screenshot, top_left, bottom_right, (0, 255, 0), 2)
        cv.imshow("Matched Result", screenshot)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Object not found.")
