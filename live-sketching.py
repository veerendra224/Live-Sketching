import cv2
import numpy as np

# Our sketch generating function
def sketch(image):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Clean up image using Gaussian Blur
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Extract edges
    canny_edges = cv2.Canny(img_gray_blur, 20, 50)
    
    # Do an invert binarize the image 
    _, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask

# Initialize webcam, cap is the object provided by VideoCapture
# It contains a boolean indicating if it was successful (ret)
# It also contains the images collected from the webcam (frame)
cap = cv2.VideoCapture(0)

# Set desired laptop screen dimensions
desired_width = 1450
desired_height = 768

while True:
    ret, frame = cap.read()
    
    # Resize frame to desired laptop screen dimensions
    frame = cv2.resize(frame, (desired_width, desired_height))
    
    cv2.imshow('Our Live Sketcher', sketch(frame))
    if cv2.waitKey(1) == 13: # 13 is the Enter Key
        break
        
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
