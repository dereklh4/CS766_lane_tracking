# import libraries
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats

# this function creates the training data for class c for frame t.
# k = number of previous frames (default set arbitrarily to 10)
# list_of_RGB_Images = input images
# imagesMasks = correspond to either yellow lane vs. road or white lane vs. road
def createTrainingData(list_of_RGB_Images, imageMasks_w, imageMasks_y):
    # Create a list of all RGB values
    R = []
    G = []
    B = []
    for img in list_of_RGB_Images:
        # reshape each color val into an array
        currR = img[:,:,0].reshape(np.shape(img[:,:,0])[0]*np.shape(img[:,:,0])[1]) 
        currG = img[:,:,1].reshape(np.shape(img[:,:,1])[0]*np.shape(img[:,:,1])[1])
        currB = img[:,:,2].reshape(np.shape(img[:,:,2])[0]*np.shape(img[:,:,2])[1])
        
        # concatenate with R, G, and B
        R = np.concatenate((R, currR))
        G = np.concatenate((G, currG))
        B = np.concatenate((B, currB))
        
    y_w = []
    for mask in imageMasks_w:
        # for both white/yellow, there should be a high R value, so just take
        # red value, reshape, then set to binary
        currY = mask[:,:,0].reshape(np.shape(mask)[0]*np.shape(mask)[1]) 
        
        # convert currY to binary (0 = road, 1 = lane)
        currY = np.where(currY > 0, 1, 0)
        
        y_w = np.concatenate((y_w, currY))
        
    y_y = []
    for mask in imageMasks_y:
        # for both white/yellow, there should be a high R value, so just take
        # red value, reshape, then set to binary
        currY = mask[:,:,0].reshape(np.shape(mask)[0]*np.shape(mask)[1]) 
        
        # convert currY to binary (0 = road, 1 = lane)
        currY = np.where(currY > 0, 1, 0)
        
        y_y = np.concatenate((y_y, currY))
        
    # create training data to return
    X = np.zeros((len(R), 3))
    X[:,0] = R
    X[:,1] = G
    X[:,2] = B
    
    return X, y_w, y_y

# this function fits data using LDA to find conversion vector from previous images (X,y)
# then applies learned weights to current image (currImage)
def applyLDA(X, y):
    # create classifier object
    clf = LinearDiscriminantAnalysis()

    # fit classifier to data
    clf.fit(X, y)
    
    # return learned weights (i.e. the gradient-enhancing conversion vector)
    return clf.coef_

# this function converts an image with the calculated gradient-enhancing vector
def convertToGray(w, img):
    grayImg = np.dot(img, w)
    #grayImg = (w[0]*img[:,:,0] + w[1]*img[:,:,1] + w[2]*img[:,:,2]).astype(np.uint8)
    return grayImg

# this function reads in the initial images and their resepective masks
def readInitImages():
    rgbImages = []
    masks_w = []
    masks_y = []
    imgNum = np.arange(1,6)   
    for count in imgNum:
        currImg = cv2.imread('laneData/img'+str(count)+'.jpg')
        whiteMask = cv2.imread('laneData/img'+str(count)+'_lane_w.jpg')
        yellowMask = cv2.imread('laneData/img'+str(count)+'_lane_y.jpg')
        
        # add to list to return
        rgbImages.append(currImg)
        masks_w.append(whiteMask)
        masks_y.append(yellowMask)
    
    return rgbImages, masks_w, masks_y

# read in initial 5 images with respective masks
list_of_RGB_Images, imageMasks_w, imageMasks_y = readInitImages()

# create training data
X, y_w, y_y = createTrainingData(list_of_RGB_Images, imageMasks_w, imageMasks_y)

# for each mask, compute LDA
colorMask = [y_w, y_y]
gradientEnhancedVectors = []
for mask in colorMask:
    weight = applyLDA(X, mask)
    gradientEnhancedVectors.append(weight)

# read in image with masks
img = cv2.imread('laneData/img6.jpg')

 # test on sixth image and save result
colorConv = ['w','y']
count = 0
for w in gradientEnhancedVectors:
    # convert image to gray with weight for respective class (w, then y)
    w = w[0]
    grayImg = convertToGray(w, img)
    
    # save image
    cv2.imwrite("laneData/img6_gray_" + str(colorConv[count]) + ".jpg", grayImg)
    
    # display image
    plt.imshow(grayImg, cmap = plt.get_cmap('gray'))
    plt.show()
    count = count + 1







   
