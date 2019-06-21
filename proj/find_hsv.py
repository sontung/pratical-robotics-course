import cv2
import numpy as np

color = np.uint8([[[151, 120, 77]]])
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
print(hsv)

color = np.uint8([[[0, 0, 141]]])
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
print(hsv)

color = np.uint8([[[0, 142, 73]]])
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
print(hsv)
