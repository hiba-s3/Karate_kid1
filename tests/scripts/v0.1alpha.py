import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

template_paths = [
    r'CV_final_project\assets\templates\fan.jpg',
    r'CV_final_project\assets\templates\fblue.jpg'
]
image_path = r'CV_final_project\history\1.jpg'

target_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
matched_image = target_image.copy()

threshold = 0.9
image_width = target_image.shape[1]
total_matches = 0
results = {}

for idx, template_path in enumerate(template_paths):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Failed to load template: {template_path}")
        continue
    result = cv2.matchTemplate(target_image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    locations_list = list(zip(*locations[::-1]))
    num_matches = len(locations_list)
    total_matches += num_matches
    h, w = template.shape
    for loc in locations_list:
        top_left = loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        y1, y2 = top_left[1], bottom_right[1]
        
        position = "Left" if top_left[0] < image_width / 2 else "Right"
        color = (255, 0, 0) if position == "Left" else (0, 255, 0)
        
        cv2.rectangle(matched_image, top_left, bottom_right, color, 2)
        
        if idx == 2 and num_matches > 0:
            results['fang'] = position
            results['heightg'] = y2 - y1
            results['yg'] = y2
        if idx == 5 and num_matches > 0:
            results['fanb'] = position
            results['heightb'] = y2 - y1
            results['yb'] = y2


height, width, _ = image.shape

crop_width = int(width * 0.4)  
left_boundary = crop_width
right_boundary = width - crop_width

image = image[:, left_boundary:right_boundary]

pixels = image.reshape(-1, 3)
pixels = np.float32(pixels)  
K =3
kmeans = KMeans(n_clusters=K, random_state=42)
labels = kmeans.fit_predict(pixels)
centers = kmeans.cluster_centers_  
segmented_image = centers[labels].reshape(image.shape)
segmented_image = np.uint8(segmented_image)

if cat==0: 
  normal(image)
elif cat==2:
  poscolor(segmented_image)
  num2(image,centers,labels)
elif cat==3 or cat==4:
  poscolor(segmented_image)
  num34(image,centers,labels)
  
elif cat==-1:
  azxn(image)
  
  
def normal(image):  
     
    cluster_index = 1
    highlighted_image = np.ones_like(segmented_image) * 255
    mask = labels.reshape(image.shape[:2]) == cluster_index  
  
    highlighted_image[mask] = centers[cluster_index]  
 
    target_image=highlighted_image
    image = target_image
    gray=cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    imgor1=target_image
    kernel_size = 29
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    _, binary = cv2.threshold(gray, 132, 255, cv2.THRESH_BINARY)

    dilation = cv2.dilate(binary, kernel, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(closing, 40, 80, apertureSize=3)  
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=35, minLineLength=1, maxLineGap=100)

    output_image = image.copy()
    if lines is not None: 
       for line in lines: 
        middle = output_image.shape[1] // 2
          
        x1, y1, x2, y2 = line[0]
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  
        
        line_height = abs(y2 - y1)  
        print(f"Line from ({x1}, {y1}) to ({x2}, {y2}) has height: {line_height} pixels")
        
        
        if x1 < middle and x2 < middle:
            position = "left"
        elif x1 > middle and x2 > middle:
            position = "right"
        else:
            position = "spanning both sides"
        
        print(f"The line is on the {position} side.")
        
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_image=cv2.cvtColor(output_image,cv2.COLOR_RGB2BGR)
   
       
def num2(image,centers,labels):   
    
     
    color_to_replace_index = 0 
    new_color_index = 1      

    centers[color_to_replace_index] = centers[new_color_index]  
    new_segmented_image = centers[labels].reshape(image.shape)  
    new_segmented_image = np.uint8(new_segmented_image)  

    
    pixels = new_segmented_image.reshape(-1, 3)
    pixels = np.float32(pixels)  
    K =4
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(pixels)  
    centers = kmeans.cluster_centers_  
    segmented_image1 = centers[labels].reshape(new_segmented_image.shape)
    segmented_image1 = np.uint8(segmented_image) 

    cluster_index = 0
    highlighted_image = np.ones_like(segmented_image1) * 255
    mask = labels.reshape(image.shape[:2]) == cluster_index  

    highlighted_image[mask] = centers[cluster_index]  
  
    target_image=highlighted_image
    image = target_image
    gray=cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    imgor1=target_image
    kernel_size = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    _, binary = cv2.threshold(gray, 132, 255, cv2.THRESH_BINARY)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    kernel_size = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    kernel_size = 42
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilation = cv2.dilate(opening, kernel, iterations=1)
   
    
    edges = cv2.Canny(dilation, 40, 80, apertureSize=3)  
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=1, maxLineGap=100)

    output_image = image.copy()
    if lines is not None:   
      for line in lines:
        middle = output_image.shape[1] // 2
          
        x1, y1, x2, y2 = line[0]
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  
        
        line_height = abs(y2 - y1)  
        print(f"Line from ({x1}, {y1}) to ({x2}, {y2}) has height: {line_height} pixels")
        
        
        if x1 < middle and x2 < middle:
            position = "left"
        elif x1 > middle and x2 > middle:
            position = "right"
        else:
            position = "spanning both sides"
        
        print(f"The line is on the {position} side.")
        
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
         

    output_image=cv2.cvtColor(output_image,cv2.COLOR_RGB2BGR)
    
def num34(image,centers,labels):   
    
     
    color_to_replace_index = 0 
    new_color_index = 1      

    centers[color_to_replace_index] = centers[new_color_index]  
 
    new_segmented_image = centers[labels].reshape(image.shape)  
    new_segmented_image = np.uint8(new_segmented_image)  
 

    centers[color_to_replace_index] = centers[new_color_index]  

    new_segmented_image = centers[labels].reshape(image.shape)  
    new_segmented_image = np.uint8(new_segmented_image)  


    

    pixels = new_segmented_image.reshape(-1, 3)
    pixels = np.float32(pixels)  
    K =4
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(pixels)  
    centers = kmeans.cluster_centers_  
    segmented_image1 = centers[labels].reshape(new_segmented_image.shape)
    segmented_image1 = np.uint8(segmented_image) 

    cluster_index = 0
    highlighted_image = np.ones_like(segmented_image1) * 255
    mask = labels.reshape(image.shape[:2]) == cluster_index  

    highlighted_image[mask] = centers[cluster_index]  
  
    target_image=highlighted_image
    image = target_image
    gray=cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    imgor1=target_image
    kernel_size = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    _, binary = cv2.threshold(gray, 132, 255, cv2.THRESH_BINARY)

    dilation = cv2.dilate(binary, kernel, iterations=1)


    kernel_size = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    kernel_size = 24
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilation = cv2.dilate(opening, kernel, iterations=1)
   
    
    edges = cv2.Canny(dilation, 40, 80, apertureSize=3)  
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=1, maxLineGap=100)

    output_image = image.copy()
    if lines is not None:   
      for line in lines:
        middle = output_image.shape[1] // 2
          
        x1, y1, x2, y2 = line[0]
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  
        
        line_height = abs(y2 - y1)  
        print(f"Line from ({x1}, {y1}) to ({x2}, {y2}) has height: {line_height} pixels")
        
        
        if x1 < middle and x2 < middle:
            position = "left"
        elif x1 > middle and x2 > middle:
            position = "right"
        else:
            position = "spanning both sides"
        
        print(f"The line is on the {position} side.")
        
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
         

    output_image=cv2.cvtColor(output_image,cv2.COLOR_RGB2BGR)
    


 
  

    
def azxn(image) : 
    if image.shape[2] == 3:  
        
        b, g, r = cv2.split(image)  
        a = None   
    elif image.shape[2] == 4:    
        b, g, r, a = cv2.split(image)    
    else:  
        print("Unexpected number of channels")  
        exit()  

    b[:] = 0
    r[:] = 150
    
    g[:] -= 210
    if a is not None: 
        
        modified_image = cv2.merge((b, g, r, a))  
    else:    
        modified_image = cv2.merge((b, g, r))  

    modified_image_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)  

    
    pixels = modified_image_rgb.reshape(-1, 3)
    pixels = np.float32(pixels)  
    K =3
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(pixels)  
    centers = kmeans.cluster_centers_  
    segmented_image = centers[labels].reshape(modified_image_rgb.shape)
    segmented_image = np.uint8(segmented_image) 



    color_to_replace_index = 0   
    new_color_index = 1     

    centers[color_to_replace_index] = centers[new_color_index]  

    new_segmented_image = centers[labels].reshape(image.shape)  
    new_segmented_image = np.uint8(new_segmented_image)  

    centers[color_to_replace_index] = centers[new_color_index]  

    new_segmented_image = centers[labels].reshape(image.shape)  
    new_segmented_image = np.uint8(new_segmented_image)  

   

    pixels = new_segmented_image.reshape(-1, 3)
    pixels = np.float32(pixels)  
    K =4
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(pixels)  
    centers = kmeans.cluster_centers_  
    segmented_image1 = centers[labels].reshape(new_segmented_image.shape)
    segmented_image1 = np.uint8(segmented_image) 

    cluster_index = 0
    highlighted_image = np.ones_like(segmented_image1) * 255
    mask = labels.reshape(image.shape[:2]) == cluster_index  

    highlighted_image[mask] = centers[cluster_index]   

   
    target_image=highlighted_image
    image = target_image
    gray=cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    imgor1=target_image
    kernel_size = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    _, binary = cv2.threshold(gray, 132, 255, cv2.THRESH_BINARY)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    kernel_size = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    kernel_size = 42
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilation = cv2.dilate(opening, kernel, iterations=1)
    edges = cv2.Canny(dilation, 40, 80, apertureSize=3)  
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=1, maxLineGap=100)

    output_image = image.copy()
    if lines is not None:
    
    
        for line in lines:
            
            x1, y1, x2, y2 = line[0]
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  

    output_image=cv2.cvtColor(output_image,cv2.COLOR_RGB2BGR)
    
def poscolor(segmented_image):
        
    original_image = segmented_image

    color_1 = centers[0]  

    lower_bound = np.array(color_1) - 30    
    upper_bound = np.array(color_1) + 30  

    lower_bound = lower_bound[::-1]  
    upper_bound = upper_bound[::-1]  

    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  

    mask = cv2.inRange(rgb_image, lower_bound, upper_bound)  

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    for contour in contours:  
        
        if cv2.contourArea(contour) > 100: 
            x, y, w, h = cv2.boundingRect(contour)  
            yofcolor=y
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 255), 2)    
