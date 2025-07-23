import numpy as np
import cv2
def detect_lines(image):
    
    
    _, binary_image = cv2.threshold(image, 16, 255, cv2.THRESH_BINARY)
    
    
    # blurred_image = cv2.GaussianBlur(binary_image, (3, 3), 0)
    blurred_image = binary_image
    edges = cv2.Canny(blurred_image, 25, 120, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=2, minLineLength=2, maxLineGap=5)
    
    debug_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)  
    
    
    horizontal_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            
            angle_threshold = np.pi / 4  
            angle = np.arctan2(y2 - y1, x2 - x1)
            
            if abs(angle) < angle_threshold:  
                horizontal_lines.append(line)  
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  
    else:
        print("No lines detected")

    return horizontal_lines , debug_image 
def determine_position(image_shape, lines):
    direction = "right"
    if lines is None:
        return direction
    y = 0
    middle = image_shape[1] // 2
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y < y1 and y < y2:
            if x1 < middle and x2 < middle:
                
                direction = "right"
            elif x1 > middle and x2 > middle:
                
                direction = "left"
            y = min(y1, y2)
    return direction
def process(image, count, temp_glass=False, glass=False):
    
    # if len(image.shape) == 3:  
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    lines, debug_image = detect_lines(image)
    position = determine_position(image.shape, lines)
    
    cv2.imwrite(f".\\history\\{count}_{position}_debug.png", debug_image)
    return position

def line_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
def filter_long_lines(line,max_line_length):
        x1, y1, x2, y2 = line[0]
        if line_length(x1, y1, x2, y2) <= max_line_length:
            return True
        return False

def detect_lines_with_glass(image):
    
    
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    edges = cv2.Canny(blurred_image, 25, 125, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=2, maxLineGap=2)
    
    debug_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)  
    
    
    
    horizontal_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            short_line = filter_long_lines(line,30)
            
            angle_threshold = np.pi / 4  
            angle = np.arctan2(y2 - y1, x2 - x1)
            
            if abs(angle) < angle_threshold and short_line:  
                horizontal_lines.append(line)  
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  
    else:
        print("No lines detected")
    return horizontal_lines , debug_image 

def detect_glass(image,count):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray_image, 50, 150)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=50, maxLineGap=10)
    
    horizontal_line_detected = False
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if abs(y1 - y2) < 10:  
                horizontal_line_detected = True
                
    if horizontal_line_detected:
        print(f"Horizontal line detected. Glass likely present at step {count}")
    else:
        
        pass
    return horizontal_line_detected

def detectIFisnum (images):
    
    num=0
    
    image = images
    b, g, r = cv2.split(image)
    
    if(True) :
        b_modified = b - 124
        r_modified = r - 91
        g_modified = g - 98
        modified_image = cv2.merge((b_modified, g_modified, r_modified))

        target_color = np.array([0, 0, 0])  

        tolerance = 10
        lower_bound = target_color - tolerance
        upper_bound = target_color + tolerance

        lower_bound = np.clip(lower_bound, 0, 255)
        upper_bound = np.clip(upper_bound, 0, 255)

        mask = cv2.inRange(modified_image, lower_bound, upper_bound)

        if cv2.countNonZero(mask) > 0:
          
        
            
            num=4
            b_modified = b - 205
            r_modified = r - 172
            g_modified = g - 189
            modified_image = cv2.merge((b_modified, g_modified, r_modified))
            target_color = np.array([0, 0, 0])  
            tolerance = 10
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            lower_bound = np.clip(lower_bound, 0, 255)
            upper_bound = np.clip(upper_bound, 0, 255)

            mask2 = cv2.inRange(modified_image, lower_bound, upper_bound)
            if cv2.countNonZero(mask2) > 0:
                
                
                num=40 
            return num
      
       
    if(True) :
        
        
       
       
        b_modified = b - 127
        r_modified = r - 126
        g_modified = g - 109
        modified_image = cv2.merge((b_modified, g_modified, r_modified))
        target_color = np.array([0, 0, 0])  
        tolerance = 10
        lower_bound = target_color - tolerance
        upper_bound = target_color + tolerance

        lower_bound = np.clip(lower_bound, 0, 255)
        upper_bound = np.clip(upper_bound, 0, 255)

        mask = cv2.inRange(modified_image, lower_bound, upper_bound)

        if cv2.countNonZero(mask) > 0:
          
            

            num=2
            b_modified = b - 159
            r_modified = r - 142
            g_modified = g - 141
            modified_image = cv2.merge((b_modified, g_modified, r_modified))
            target_color = np.array([0, 0, 0])  
            tolerance = 10
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            lower_bound = np.clip(lower_bound, 0, 255)
            upper_bound = np.clip(upper_bound, 0, 255)

            mask2 = cv2.inRange(modified_image, lower_bound, upper_bound)
            if cv2.countNonZero(mask2) > 0:
              
             
                
                num=20
          
            return num 
      
    

    if(True) :
       
       b_modified = b - 104
       r_modified = r - 245
       g_modified = g - 225
       modified_image = cv2.merge((b_modified, g_modified, r_modified))
       target_color = np.array([0, 0, 0])  
       tolerance = 10
       lower_bound = target_color - tolerance
       upper_bound = target_color + tolerance

       lower_bound = np.clip(lower_bound, 0, 255)
       upper_bound = np.clip(upper_bound, 0, 255)

       mask = cv2.inRange(modified_image, lower_bound, upper_bound)

       if cv2.countNonZero(mask) > 0:
          
           
          
            num=3
            return num
       b_modified = b - 128
       r_modified = r - 233
       g_modified = g - 223
       modified_image = cv2.merge((b_modified, g_modified, r_modified))
       target_color = np.array([0, 0, 0])  
       tolerance = 10
       lower_bound = target_color - tolerance
       upper_bound = target_color + tolerance

       lower_bound = np.clip(lower_bound, 0, 255)
       upper_bound = np.clip(upper_bound, 0, 255)

       mask2 = cv2.inRange(modified_image, lower_bound, upper_bound)
       if cv2.countNonZero(mask2) > 0:
           
           
            num=30 
            
            return num
    if(True) :
       
        b_modified = b - 65
        r_modified = r - 177
        g_modified = g - 110
        modified_image = cv2.merge((b_modified, g_modified, r_modified))
        target_color = np.array([0, 0, 0])  
        tolerance = 10
        lower_bound = target_color - tolerance
        upper_bound = target_color + tolerance

        lower_bound = np.clip(lower_bound, 0, 255)
        upper_bound = np.clip(upper_bound, 0, 255)

        mask = cv2.inRange(modified_image, lower_bound, upper_bound)

        if cv2.countNonZero(mask) > 0:
          
           
          
            num=100
      
    return num
      
      
def prepro(image,numb):
    if numb==4 or numb==2 or numb==20 or numb==40 :
    
        if image.shape[2] == 3: 
        
     
            b, g, r = cv2.split(image)  
            a = None  
        elif image.shape[2] == 4:
     
            b, g, r, a = cv2.split(image) 
        b[:] = 0
        r[:] += 80
        r[r < 20]+=30
        g[:] = 0

        if a is not None:  
     
            modified_image = cv2.merge((b, g, r, a))  
        else:  
      
            modified_image = cv2.merge((b, g, r))  
        lower_bound = np.array([0, 0, 0])       
        upper_bound = np.array([70, 70, 70])   

        dark_mask = cv2.inRange(modified_image, lower_bound, upper_bound)
        modified_image[dark_mask > 0] = [0, 0, 200]  
 
        modified_image1 =cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB) 
        img_gray = cv2.cvtColor(modified_image1, cv2.COLOR_RGB2GRAY)
        
        kernel_size = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        _, binary = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
        dilation = cv2.dilate(binary, kernel, iterations=1)
        kernel_size = 4
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        _, binary = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)

        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)   
        #closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        #return modified_image1
        return closing  
    
    if  numb==0 or numb==3 or numb==30 or numb==100:
    
        if image.shape[2] == 3: 
        
     
            b, g, r = cv2.split(image)  
            a = None  
        elif image.shape[2] == 4:
     
            b, g, r, a = cv2.split(image)  
 
        b[:] = 0
        r[:] -= 200
        g[:] = 0

        if a is not None:  
     
            modified_image = cv2.merge((b, g, r, a))  
        else:  
      
            modified_image = cv2.merge((b, g, r))  
        lower_bound = np.array([0, 0, 0])       
        upper_bound = np.array([50, 50, 50])   

        dark_mask = cv2.inRange(modified_image, lower_bound, upper_bound)
        modified_image[dark_mask > 0] = [0, 0, 200]  
 
        modified_image1 = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(modified_image1, cv2.COLOR_RGB2GRAY)
        
        kernel_size = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        _, binary = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
        dilation = cv2.dilate(binary, kernel, iterations=1)
        kernel_size = 4
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        _, binary = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)  
        #return modified_image1
        return closing    

    if numb==0 or numb==3 or numb==30 or numb==100:
    
        if image.shape[2] == 3: 
        
     
            b, g, r = cv2.split(image)  
            a = None  
        elif image.shape[2] == 4:
     
            b, g, r, a = cv2.split(image)  
 
        b[:] = 0
        r[:] -= 220
        g[:] = 0

        if a is not None:  
     
            modified_image = cv2.merge((b, g, r, a))  
        else:  
      
            modified_image = cv2.merge((b, g, r))  
        lower_bound = np.array([0, 0, 0])       
        upper_bound = np.array([50, 50, 50])   

        dark_mask = cv2.inRange(modified_image, lower_bound, upper_bound)
        modified_image[dark_mask > 0] = [0, 0, 200]  
 
        modified_image1 = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
        return modified_image1   


def calculate_white_pixel_rate(image_path):
    
    image = image_path

    
    b, g, r = cv2.split(image)

    
    white_mask = (r > 160) & (b < 60) |(g< 100)

    
    result_image = image.copy()
    result_image[white_mask] = [255, 255, 255]

    
    white_pixel_count = np.sum(white_mask)
    total_pixel_count = image.shape[0] * image.shape[1]
    white_pixel_rate = (white_pixel_count / total_pixel_count) * 100

    return white_pixel_rate
