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

    return [np.abs(clf.coef_),clf.means_]

# this function converts an image with the calculated gradient-enhancing vector
def convertToGray(w, img):
    grayImg = np.dot(img, w)

    #scale between 0 and 255
    img_vals = grayImg.flatten()
    grayImg = ((grayImg - min(img_vals)) / (max(img_vals) - min(img_vals))) * 255

    #grayImg = (w[0]*img[:,:,0] + w[1]*img[:,:,1] + w[2]*img[:,:,2]).astype(np.uint8)
    return np.uint8(grayImg)

# this function reads in the initial images and their respective masks
def readInitImages(crop_pct=.4):
    rgbImages = []
    masks_w = []
    masks_y = []
    imgNum = np.arange(1,6)   
    for count in imgNum:
        currImg = cv2.imread('laneData/img'+str(count)+'.jpg')
        whiteMask = cv2.imread('laneData/img'+str(count)+'_lane_w.jpg')
        yellowMask = cv2.imread('laneData/img'+str(count)+'_lane_y.jpg')
        
        # crop y axis
        currImg = currImg[int(480*crop_pct):np.shape(currImg)[0],:]
        whiteMask = whiteMask[int(480*crop_pct):np.shape(whiteMask)[0],:]
        yellowMask = yellowMask[int(480*crop_pct):np.shape(yellowMask)[0],:]
        
        # add to list to return
        rgbImages.append(currImg)
        masks_w.append(whiteMask)
        masks_y.append(yellowMask)
    
    return rgbImages, masks_w, masks_y

def cov(a,b):
    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum / (len(a) - 1.0)

def gaussian_intersection_solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

# read in initial 5 images with respective masks
crop_pct = .4
list_of_RGB_Images, imageMasks_w, imageMasks_y = readInitImages(crop_pct)

# create training data
X, y_w, y_y = createTrainingData(list_of_RGB_Images, imageMasks_w, imageMasks_y)

# for each mask, compute LDA
colorMask = [y_w, y_y]
gradientEnhancedVectors = []
all_class_means = []
for mask in colorMask:
    weight, class_means = applyLDA(X, mask)
    gradientEnhancedVectors.append(weight)

    #save class means. 1st for road, 2nd for lane. Note that we get a mean for R,G, and B for each class.
    road_mean = class_means[0,:]
    lane_mean = class_means[1,:]
    all_class_means.append((road_mean,lane_mean))

# read in image with masks
img = cv2.imread('laneData/img6.jpg')

 # test on sixth image and save result
colorConv = ['w','y']
count = 0
canny_img = None
for i in xrange(2):
    w = gradientEnhancedVectors[i][0]
    # convert image to gray with weight for respective class (w, then y)
    grayImg = convertToGray(w, img)
    # save image
    cv2.imwrite("laneData/img6_gray_" + str(colorConv[count]) + ".jpg", grayImg)
    
    # display image
    plt.imshow(grayImg, cmap = plt.get_cmap('gray'))
    plt.show()
    count = count + 1

    road_means, lane_means = all_class_means[i]

    #### Adaptive Canny Edge ###

    #calculcate d
    #look at last gray scale image, and get values for each mask
    mask = colorMask[i]
    last_img_mask = mask[-int(grayImg.shape[0]*grayImg.shape[1]*crop_pct):]
    last_gray_img = convertToGray(w,list_of_RGB_Images[-1])
    last_gray_img = last_gray_img.flatten()[-int(grayImg.shape[0] * grayImg.shape[1] * crop_pct):]

    lane_indices = np.nonzero(last_img_mask)
    lane_values = last_gray_img[lane_indices]
    road_indices = np.where(last_img_mask == 0)[0]
    road_values = last_gray_img[road_indices]

    lane_mean = np.mean(lane_values)
    road_mean = np.mean(road_values)

    print("Lane mean: " + str(lane_mean))
    print("Road mean: " + str(road_mean))

    lane_cov = cov(lane_values,lane_values)
    road_cov = cov(road_values,road_values)

    intersections = gaussian_intersection_solve(lane_mean,road_mean,np.sqrt(lane_cov),np.sqrt(road_cov))
    intersections = filter(lambda x: x > 0, intersections)
    d = filter(lambda x: (x > lane_mean and x < road_mean) or (x < lane_mean and x > road_mean),intersections)
    print("d:" + str(d))

    #road_weight_means = all_class_means[i][0]
    #lane_weight_means = all_class_means[i][1]
    #th_large = abs(np.dot(w,lane_weight_means) - np.dot(w,road_weight_means))
    #th_small = max(abs(np.dot(w,lane_weight_means)-d), abs(np.dot(w,road_weight_means)-d))

    #th_large = abs(lane_mean - road_mean)
    #th_small = max((lane_mean - d), abs(road_mean - d))

    th_small = road_mean
    th_large = lane_mean

    print("Large threshold: " + str(th_large))
    print("Small threshold: " + str(th_small))

    sub_canny_img = cv2.Canny(grayImg,th_small,th_large)
    plt.imshow(sub_canny_img,cmap="gray")
    plt.show()

    if canny_img is None:
        canny_img = sub_canny_img
    else:
        canny_img = canny_img | sub_canny_img

### Hough Transform with final canny image ###

plt.imshow(canny_img,cmap="gray")
plt.show()

# ??? How to specify these parameters ??? NEED TO TUNE THESE
rho = 1 #distance resolution in pixels
theta = np.pi/180 #angle resolution of accumulator in radians
threshold = 120
minimum_line_length = 150 #a line has to be at least this long
maximum_line_gap = 250 #maximum allowed gap between line segments to treat them as a single line
#Based on Robust Detection of Lines Using the Progressive Probabilistic Hough Transform by Matas, J. and Galambos, C. and Kittler, J.V.
lines = cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minimum_line_length, maximum_line_gap)

#draw the lines on a new image
line_img = img.copy()
try:
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),3)
except:
    pass
line_img = cv2.cvtColor(line_img,cv2.COLOR_BGR2RGB)
plt.imshow(line_img)
plt.show()