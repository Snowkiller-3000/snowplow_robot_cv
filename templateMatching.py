import numpy as np
import imutils
import cv2

OVERLAP_THRESHOLD = 15

def templateMatching(image, template, rectangles, tW, tH):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        
        rectangles.append((int(maxLoc[0] * r), int(maxLoc[1] * r),
                        int((maxLoc[0] + tW) * r),  int((maxLoc[1] + tH) * r)))
        
def doesOverlap(box1, box2):
   startX_1, startY_1, endX_1, endY_1 = box1
   startX_2, startY_2, endX_2, endY_2 = box2

   X_CONDITION = (startX_2 >= startX_1 or abs(startX_2 - startX_1) <= OVERLAP_THRESHOLD) and (endX_2 <= endX_1 or abs(endX_2-endX_1) <= OVERLAP_THRESHOLD)

   Y_CONDITION = (startY_2 >= startY_1 or abs(startY_2 - startY_1) <= OVERLAP_THRESHOLD) and (endY_2 <= endY_1 or abs(endY_2-endY_1) <= OVERLAP_THRESHOLD)

   if X_CONDITION and Y_CONDITION:
       return True

   return False

def mergeBoxes(rectangles):
    rectangles.sort()

    for i in range(len(rectangles)):

        box1 = rectangles[i]
        for j in range(len(rectangles)):

            box2 = rectangles[j]

            if doesOverlap(box1, box2):
                rectangles[j] = (min(box1[0],box2[0]),min(box1[1],box2[1]),max(box1[2],box2[2]),max(box1[3],box2[3]))

    rectangles = sorted(list(set(rectangles)))
    return rectangles


def buildBoundingBoxes(img, templatePath = 'assets/template.png', visualize = True):
    
    image = img 
    template = cv2.imread(templatePath)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]

    rectangles = []
    templateMatching(image, template, rectangles, tW, tH)
    mergeBoxes(rectangles)

    return rectangles