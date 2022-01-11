# import the necessary packages
import numpy as np
import imutils
import cv2

# load the image image, convert it to grayscale, and detect edges
template = cv2.imread('check.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)

rectangles = []

# loop over the images to find the template in
# load the image, convert it to grayscale, and initialize the
# bookkeeping variable to keep track of the matched region
image = cv2.imread('images/check.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None
# loop over the scales of the image
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
    r = gray.shape[1] / float(resized.shape[1])
# if the resized image is smaller than the template, then break
# from the loop
    if resized.shape[0] < tH or resized.shape[1] < tW:
        break
    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image
    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    # check to see if the iteration should be visualized
    # draw a bounding box around the detected region
    clone = np.dstack([edged, edged, edged])
    rectangles.append((int(maxLoc[0] * r), int(maxLoc[1] * r),
                      int((maxLoc[0] + tW) * r),  int((maxLoc[1] + tH) * r)))

image = cv2.imread('images/check.png')

# remove duplicates
# how do I know if a box exists in another box

# conditions for overlap:
# 1) startX box2 > starX box1 AND endX box2 > endX box1
overlapThreshold = 15


def doesOverlap(box1, box2):
   startX_1, startY_1, endX_1, endY_1 = box1
   startX_2, startY_2, endX_2, endY_2 = box2

   X_CONDITION = (startX_2 >= startX_1 or abs(startX_2 - startX_1) <= overlapThreshold) and (endX_2 <= endX_1 or abs(endX_2-endX_1) <= overlapThreshold)

   Y_CONDITION = (startY_2 >= startY_1 or abs(startY_2 - startY_1) <= overlapThreshold) and (endY_2 <= endY_1 or abs(endY_2-endY_1) <= overlapThreshold)

   if X_CONDITION and Y_CONDITION:
       return True

   return False


rectangles.sort()

for i in range(len(rectangles)):
    # CAN I MERGE RECTANGLE 1 INTO ANYTHING

    box1 = rectangles[i]
    for j in range(len(rectangles)):

        box2 = rectangles[j]

        if doesOverlap(box1, box2):
            rectangles[j] = (min(box1[0],box2[0]),min(box1[1],box2[1]),max(box1[2],box2[2]),max(box1[3],box2[3]))

rectangles = sorted(list(set(rectangles)))
print(rectangles)

for startX, startY, endX, endY in rectangles:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
cv2.waitKey(0)
