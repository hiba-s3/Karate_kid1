import cv2 as cv
import os
import datetime 
import numpy as np
def clean_folder ():
    save_location=".\\history"
    
    for filename in os.listdir(save_location):
        file_path = os.path.join(save_location, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
def isolate_trunk(image, count ):
    save_location=".\\history"
    os.makedirs(save_location, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%d%H%M%S%f")[:-3]
    
    
    height, width, _ = image.shape
    
    crop_width = int(width * 0.43)  
    left_boundary = crop_width
    right_boundary = width - crop_width - int(crop_width*0.02)
    
    crop_height = int(height * 0.57)
    down_boundary = height - crop_height
    up_boundary = int(0.36 * crop_height)
    
    cropped_image = image[up_boundary:down_boundary, left_boundary:right_boundary]
    save_path = os.path.join(save_location, f"{count}.png")
    cv.imwrite(save_path, cropped_image)
    
    
    return cropped_image

def isolate_first_trunk(image, count ):
    save_location=".\\history"
    os.makedirs(save_location, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%d%H%M%S%f")[:-3]
    
    
    height, width, _ = image.shape
    
    crop_width = int(width * 0.43)  
    left_boundary = crop_width
    right_boundary = width - crop_width - int(crop_width*0.02)
    
    crop_height = int(height * 0.47)
    down_boundary = height - crop_height
    up_boundary = int(0.5 * crop_height)
    
    cropped_image = image[up_boundary:down_boundary, left_boundary:right_boundary]
    save_path = os.path.join(save_location, f"{count}.png")
    cv.imwrite(save_path, cropped_image)
    
    
    return cropped_image


def cut_top_half(image):
    height = image.shape[0]
    return image[height // 2 - int(height * 0.1 ):, :]

def cut_down_half(image):
    height = image.shape[0]
    return image[:height // 2 + int(height * 0.1 ), :]

def horizontally_cropped_image(image ):
    save_location=".\\history"
    os.makedirs(save_location, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%d%H%M%S%f")[:-3]
    
    
    height, width, _ = image.shape
    
    crop_height = int(height * 0.57)
    down_boundary = height - crop_height
    up_boundary = int(0.36 * crop_height)
    
    cropped_image = image[up_boundary:down_boundary, :]
    # save_path = os.path.join(save_location, f"{count}.png")
    # cv.imwrite(save_path, cropped_image)
    cropped_image = cut_down_half(cropped_image)
    return cropped_image

def lantern_detection(image,count):
    template_paths = [
        r'.\\assets\\templates\\fan.jpg',
        r'.\\assets\\templates\\fblue.jpg'
    ]

    # Crop the image horizontally
    target_image = horizontally_cropped_image(image)
    target_image = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
    # Define the matching threshold
    threshold = 0.9

    image_width = target_image.shape[1]
    total_matches = 0

    fang, fanb = None, None

    for template_path in template_paths:
        # Load the template in grayscale
        template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
        if template is None:
            print(f"Error: Unable to load template {template_path}")
            continue

        # Perform template matching
        result = cv.matchTemplate(target_image, template, cv.TM_CCOEFF_NORMED)

        # Find locations exceeding the threshold
        locations = np.where(result >= threshold)
        locations_list = list(zip(*locations[::-1]))
        num_matches = len(locations_list)

        total_matches += num_matches

        # Get the template dimensions
        h, w = template.shape

        for loc in locations_list:
            top_left = loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # Determine position (Left/Right) based on image width
            position = "Left" if top_left[0] < image_width / 2 else "Right"

            # Specific conditions for the templates
            if template_path == template_paths[0] and num_matches > 0:
                fang = position
                print(f"there is a green lantern on the {position} at move number {count}")

            if template_path == template_paths[1] and num_matches > 0:
                fanb = position
                print(f"there is a blue lantern on the {position} at move number {count}")
                
    return fanb if fanb else fang


def detect_game_over(full_screen, template_paths=[".\\assets\\templates\\game_over.png", ".\\assets\\templates\\out_of_energy.png"]):
    full_screen_gray = cv.cvtColor(full_screen, cv.COLOR_BGR2GRAY)
    
    for template_path in template_paths:
        template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
        if template is None:
            print(f"The template image at {template_path} was not found!")
            continue  
        
        result = cv.matchTemplate(full_screen_gray, template, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        threshold = 0.7
        
        if max_val >= threshold:
            print("Game over detected!")
            return True
    
    return False
def save_history():
    pass 