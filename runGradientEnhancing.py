# import libraries
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
import sys

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

    w = clf.coef_

    # scale between .1 and 1
    # import sklearn
    # scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(.1,1))
    # scaler = scaler.fit(w.reshape(-1,1))
    # w_scaled = scaler.transform(w.reshape(-1,1))
    #
    # w_scaled = w_scaled / np.linalg.norm(w_scaled,1)
    # w = np.transpose(w_scaled)

    w = np.abs(w)

    return [w,clf.means_]

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

def get_slope(line):
    x1, y1, x2, y2 = line[0]
    slope = ((y2-y1) / float(x2-x1))
    return slope

def filter_lines(lines,img_height,thresh_h_percentage,slope_cutoff):
    """Lines should go through both the near and far region of an image. The cutoff for near and far is determined by thresh_h.
    Also, lane slopes should be >= slope_cutoff (should probably be around .3 since they should approach vertical lines"""
    final_lines = []
    thresh_h = img_height * thresh_h_percentage
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (y1 < thresh_h and y2 > thresh_h) or (y1 > thresh_h and y2 < thresh_h):  # far near
                slope = get_slope(line)
                if abs(slope) >= slope_cutoff: #slope is likely to approach "vertical" slopes
                    final_lines.append(line)
    return final_lines

def keep_part_of_image(img,h_percentage_to_keep):
    """only keep the bottom portion of the image to feed to hough. Zero out the rest"""
    mask = np.zeros_like(img)
    h, w = img.shape[:2]
    h_keep = int(h * (1-h_percentage_to_keep))

    # keep some bottom percentage of image
    mask[h_keep:h][0:w] = 1

    masked_img = cv2.bitwise_and(img,mask)
    return masked_img

def draw_lines(img,lines):
    """returns a copy of the img with the lines drawn on it"""
    line_img = img.copy()
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    except:
        pass
    return line_img

def get_lowest_line(lines):

    #actually the "lowest" line will be the one with the highest y
    max_loc = None
    max_value = -sys.maxint
    for i in xrange(len(lines)):
        line = lines[i]
        x1,y1,x2,y2 = line[0]
        if y1 > max_value:
            max_value = y1
            max_loc = i
        if y2 > max_value:
            max_value = y2
            max_loc = i
    return [max_loc,max_value]


def select_predictions(lines,img_w):
    """Given several lines left, make the best guess as to which are the final 2"""
    final_lines = []

    line_slope_tuples = map(lambda x: (x,get_slope(x)), lines)
    pos_slope_tuples = filter(lambda x: x[1] > 0, line_slope_tuples)
    neg_slope_tuples = filter(lambda x: x[1] <= 0, line_slope_tuples)

    if len(neg_slope_tuples) == 1 and len(pos_slope_tuples) == 1: #best case. 1 for each
        return lines
    elif len(neg_slope_tuples) == 1 or len(pos_slope_tuples) == 1:
        if len(neg_slope_tuples) == 1:
            final_line = neg_slope_tuples[0][0]
            final_lines.append(neg_slope_tuples[0][0])
            lines = map(lambda x: x[0],pos_slope_tuples)
        if len(pos_slope_tuples) == 1:
            final_line = pos_slope_tuples[0][0]
            final_lines.append(pos_slope_tuples[0][0])
            lines = map(lambda x: x[0],neg_slope_tuples)

        #choose the lowest line for the other side
        i, min_val = get_lowest_line(lines)
        other_line = lines[i]

        final_lines.append(other_line)

    elif len(neg_slope_tuples) == 0 or len(pos_slope_tuples) == 0: #not getting one of the lanes...for other one, just return the one with lowest x
        min_i, min_val = get_lowest_line(lines)
        final_lines.append(lines[min_i])
    else:
        #both have more than 1

        #return the two lowest lines in each group
        pos_lines = map(lambda x: x[0], pos_slope_tuples)
        neg_lines = map(lambda x: x[0], neg_slope_tuples)

        i,min_val = get_lowest_line(pos_lines)
        j,min_val = get_lowest_line(neg_lines)

        final_lines.append(pos_lines[i])
        final_lines.append(neg_lines[j])

    return final_lines

def region_of_interest(lines):
    if len(lines) > 2:
        raise Exception("Shouldn't be calculating a region of interest for len(lines) > 2")


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

    #calc d
    # lane_cov = cov(lane_values,lane_values)
    # road_cov = cov(road_values,road_values)
    #
    # intersections = gaussian_intersection_solve(lane_mean,road_mean,np.sqrt(lane_cov),np.sqrt(road_cov))
    # intersections = filter(lambda x: x > 0, intersections)
    # d = filter(lambda x: (x > lane_mean and x < road_mean) or (x < lane_mean and x > road_mean),intersections)
    # print("d:" + str(d))

    th_small = road_mean
    th_large = lane_mean

    #print("Large threshold: " + str(th_large))
    #print("Small threshold: " + str(th_small))

    sub_canny_img = cv2.Canny(grayImg,th_small,th_large)
    plt.imshow(sub_canny_img,cmap="gray")
    plt.show()

    if canny_img is None:
        canny_img = sub_canny_img
    else:
        canny_img = canny_img | sub_canny_img

### Keep only bottom part of canny image ###

plt.imshow(canny_img,cmap="gray")
plt.show()

canny_img = keep_part_of_image(canny_img,.5)
plt.imshow(canny_img,cmap="gray")
plt.show()

### Hough Transform on canny image ###
#TODO: May need to modify these more

rho = 2 #distance resolution in pixels
theta = np.pi/180 #angle resolution of accumulator in radians
threshold = 110
minimum_line_length = 80 #a line has to be at least this long
maximum_line_gap = 250 #maximum allowed gap between line segments to treat them as a single line
#Based on Robust Detection of Lines Using the Progressive Probabilistic Hough Transform by Matas, J. and Galambos, C. and Kittler, J.V.
lines = cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minimum_line_length, maximum_line_gap)

### Filter the resulting lines ###

thresh_h_percentage = .7
slope_cutoff = .3
lines = filter_lines(lines,img.shape[0],thresh_h_percentage,slope_cutoff)
print("Num final lines: " + str(len(lines)))

#draw lines on image to see results
line_img = draw_lines(img,lines)
line_img = cv2.cvtColor(line_img,cv2.COLOR_BGR2RGB)
plt.imshow(line_img)
plt.show()

#reduce down to just 2 lines with some heuristics. In future, would be better to do heuristic of choosing lanes that best overlap with labels from last image
lines = select_predictions(lines,img.shape[1])
print("Reduced to " + str(len(lines)))

#draw lines on image to see results
line_img = draw_lines(img,lines)
line_img = cv2.cvtColor(line_img,cv2.COLOR_BGR2RGB)
plt.imshow(line_img)
plt.show()

"""Next steps: 
-keep region of interest around those lines
-collect edges in region that have similar slope as HT line as lane edges
-curve fitting to those lane edges"""

## Region of Interest ## keep a region of interest around each line for lane edges to fit
#region_of_interest(lines)