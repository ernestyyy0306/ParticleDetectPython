# import packages
import cv2
import numpy as np
from PIL import Image
from math import sqrt
from imutils import contours
from skimage import measure
import argparse
import imutils
import sys
import pandas as pd
from math import pi
 
# Picture path
img = cv2.imread('c_ref.PNG')
a = [0,0] # x-coor
b = [0,0] # y-coor
ref = 0
point = 0
pixel_ref = 0
num = []  #export data
diameter = []  #export data
radius = []
 
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global ref
        global pixel_ref
        # point out the reference
        if ref < 2:
            xy = "%d,%d" % (x, y)
            cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.imshow("image", img)
            a[ref] = x
            b[ref] = y
            pixel_ref = sqrt((a[0] - a[1]) ** 2 + (b[0] - b[1]) ** 2)
            print (f"Reference point {ref+1}")
            ref = ref + 1

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey(0)

# get the input value from user
actual_ref = int(input("The distance scale of the reference (nm) : "))
scale_percent = int(input("Sensitivity (1 to 10) : "))
threshold = int(input("Contrast of Image (0 to 255) : "))

# convert all the white background into black
for i in range (np.shape(img)[0]):
    for j in range (np.shape(img)[1]):
        if (img[i][j][2] > 80 and img[i][j][1] > 80 and img[i][j][0] > 80):
            img[i][j][2] = 0
            img[i][j][1] = 0
            img[i][j][0] = 0

#convert it to grayscale, enlarge and blur it
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray, (11, 11), 0)
width = int(gray_image.shape[1] * scale_percent)
height = int(gray_image.shape[0] * scale_percent)

# dsize
dsize = (width, height)
output = cv2.resize(gray_image, dsize)

# thresh equal to blackandwhiteimage
thresh = cv2.threshold(output, threshold, 255, cv2.THRESH_BINARY)[1]

# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

print ("Calculating......")

# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "large"
# components
#labels = measure.label(thresh, neighbors=8, background=0)
labels = measure.label(thresh, connectivity=2, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

# loop over the unique components
for label in np.unique(labels):
    # if this is the background label, ignore it
    if label == 0:
        continue
    # otherwise, construct the label mask and count the
    # number of pixels 
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    # if the number of pixels in the component is sufficiently 
    # large, then add it to our mask of "large blobs"
    if numPixels > 300:
        mask = cv2.add(mask, labelMask)

# find the contours in the mask, then sort them from left to right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]

#declaration
myDict = {}
distance = {}
distDict = {}
closestParticle = []
closestDist = []
count = 0
neighbourDist = {}
averageDist = 0
shortestDist = []
total_area = 0
sa = []

# loop over the contours
for (i, c) in enumerate(cnts):
    # draw the bright spot on the image
    (x, y, w, h) = cv2.boundingRect(c)
    x = int(x/scale_percent)
    y = int(y/scale_percent)
    w = int(w/scale_percent)
    h = int(h/scale_percent)
    ((cX, cY), pixel_radius) = cv2.minEnclosingCircle(c)
    cX = int(cX/scale_percent)
    cY = int(cY/scale_percent)
    pixel_radius = int(pixel_radius/scale_percent)
    cv2.circle(img, (int(cX), int(cY)), int(pixel_radius),
        (0, 0, 255), 3)

    # calculate diameter
    number = i+1
    actual_diameter = ((pixel_radius*2*actual_ref)/pixel_ref)
    actual_radius = actual_diameter/2
    surface = (4*pi*(actual_radius ** 2))
    sa.append(surface)
    total_area = total_area + surface
    num.append(number)
    radius.append(pixel_radius)
    diameter.append(actual_diameter)
#-------------------------------------------------------------------
    myDict[number] = [cX,cY,pixel_radius] 
#-------------------------------------------------------------------
    cv2.putText(img, "#{}".format(i + 1), (x, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)    

#-------------------------------------------------------------------------------------------------------
tempo = []
test = []
for i in range(1, len(myDict)+1):
    distance.clear()
    neighbourDist.clear()
    tempo.clear()
    for j in range(1, len(myDict)+1):
        if(i!=j):
            temp = int(sqrt((myDict[i][0] - myDict[j][0]) ** 2 + (myDict[i][1] - myDict[j][1]) ** 2))
            distance[j] = temp

    sortedDist = sorted(distance.items(), key=lambda x:x[1])
    for x in range(10):
        if sortedDist[x][1] < ( sortedDist [0][1] +  ((myDict[sortedDist[0][0]][2]) * 2) ):
            neighbourDist[sortedDist[x][0]] = sortedDist[x][1]
            shortestPart = list(neighbourDist.keys())
            shortestDist = list(neighbourDist.values())

    for y in range(len(shortestDist)):
        tempo.append(shortestPart[y]) 
        tempo.append((shortestDist[y]*actual_ref)/pixel_ref)
    distDict[i] = tempo[:]

avg = []
for g in list(distDict.keys()):
    totalDist = 0
    for h in range(int(len(distDict[g])/2)):
        totalDist = (totalDist + distDict[g][h*2+1])
    avg.append(totalDist/(len(distDict[g])/2))
    
#print(avg)
#print(distDict)

#-------------------------------------------------------------------------------------------------------

# Export to excel
df = pd.DataFrame({
    "# No" : num,
    "Particle Diameter (nm)" : diameter,
    #"Nearest Particle 1 " : closestParticle[1][0],
    "Nearest Particle Average Distance (nm)" : avg,
    "Surface Area (nm2)" : sa
    })

df.to_excel('output.xlsx', index = False)
print("Total particles: " + str(number))
average_surface = total_area/number
print("Total surface area: " + str(total_area) + " nm2")
print("Average surface area: " + str(average_surface) + ' nm2')
cv2.imshow("Image", img)
#cv2.imwrite('a_distribution.png', img) 
cv2.waitKey(0)