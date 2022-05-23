#!/usr/bin/env python
# coding: utf-8

# # Load the model

# In[1]:


import sklearn.externals as extjoblib
import joblib

# Upload the saved svm model:
model = joblib.load('LinearSVC.npy')
print(model)


# In[2]:


import cv2
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# In[3]:


from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage import color
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob
from skimage import exposure
from sklearn.preprocessing import StandardScaler


#Define HOG Parameters
# change them if necessary to orientations = 8, pixels per cell = (16,16), cells per block to (1,1) for weaker HOG
orientations = 9 
pixels_per_cell = (8, 8)
cells_per_block =(2, 2)
threshold = .3

# define the sliding window:
def sliding_window(image, stepSize, windowSize):# image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):# this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])
#%%
# Upload the saved svm model:
#model = joblib.load('Inser\Path\of_the_trained\SVM-model\here')

# Test the trained classifier on an image below!
global last_sc
global counter 
global last_rects
counter = 0
global empty_frame_counter
empty_frame_counter = 0
def process_image(img):
    global last_sc
    global counter
    global last_rects
    global empty_frame_counter
    counter += 1 
    gen_new_detc = 1
    if counter % 5 != 1 :
        img= imutils.resize(img, width=600,height=600) 
        sc = last_sc 
        rects = last_rects
        gen_new_detc = 0
        print(counter)  
    
    if gen_new_detc :
        
        print('enter')
   # image = china[150:220, 130:250]
#     img = StandardScaler(with_mean=0, with_std=1).fit_transform(img)
    #img = img / 255.0
        scale = 0
        detections = []
        # read the image you want to detect the object in:
        #img= cv2.imread("image0068.png")
        #img= cv2.imread("car_.png")
        #img= cv2.imread("straight_lines1.jpg")
        #img= cv2.imread("test1.jpg")
        #img=image_resize(img, width = int(img.shape[1]*0.4), height = int(img.shape[0]*0.4))

        # img= imutils.resize(img, width=int(img.shape[1]*0.15),height=int(img.shape[0]*0.15))
        img= imutils.resize(img, width=600,height=600) 
        # Try it with image resized if the image is too big  #**************************(32,32)
        #img= cv2.resize(64,64) # can change the size to default by commenting this code out our put in a random number

        # defining the size of the sliding window (has to be, same as the size of the image in the training data)
        (winW, winH)= (64,64)   #**************************(32,32)
        windowSize=(winW,winH)
        downscale=1.85
        # Apply sliding window:
        for resized in pyramid_gaussian(img, downscale=1.85): # loop over each layer of the image that you take!
            # loop over the sliding window for each layer of the pyramid
    #         if(scale==0):
    #             scale +=1
    #             continue

            for (x,y,window) in sliding_window(resized, stepSize=6, windowSize=(winW,winH)):

                # if the window does not meet our desired window size, ignore it!

                if window.shape[0] < winH or window.shape[1] < winW: # ensure the sliding window has met the minimum size requirement
                    continue

                if (y < resized.shape[0] / 1.9) or (x  < resized.shape[1]* 0.6) or (y > resized.shape[0] * 0.7):
                    continue

                #window=color.rgb2gray(window)

                fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')  # extract HOG features from the window captured      
                fds = fds.reshape(1, -1) # re shape the image to make a silouhette of hog

                hog_image_rescaled = exposure.rescale_intensity(fds, in_range=(0, 0.02))
                pred = model.predict(fds) # use the SVM model to make a prediction on the HOG features extracted from the window

               # window=color.gray2rgb(window)
                if pred == 'vehicles':
    #                 print(pred)
    #                 print(model.decision_function(fds))
                    if model.decision_function(fds) > 0.4 :  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
    #                     print("Detection:: Location -> ({}, {})".format(x, y))
                        print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))

                        detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                           int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                              int(windowSize[1]*(downscale**scale))))
            
            scale+=1
            if(scale==2):
                break

       # clone = resized.copy()
    #     for (x_tl, y_tl, _, w, h) in detections:
    #         print("edffffffffd")
    #         cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 1)
        rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes
        sc = [score[0] for (x, y, score, w, h) in detections]
    #     print("detection confidence score: ", sc)
        sc = np.array(sc)
        if not rects.any() and last_rects.any():
            empty_frame_counter +=1
            if  empty_frame_counter < 15:
                sc = last_sc 
                rects = last_rects
            else:
                empty_frame_counter = 0  
            
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
    last_sc = sc
    last_rects = rects
    # the peice of code above creates a raw bounding box prior to using NMS
    # the code below creates a bounding box after using nms on the detections
    # you can choose which one you want to visualise, as you deem fit... simply use the following function:
    # cv2.imshow in this right place (since python is procedural it will go through the code line by line).
    draw = []
    if len(pick) != 0:    
        flag=0
#         draw.append(pick[0])
    
        for (xA, yA, xB, yB) in pick:
            for (mA, nA, mB, nB) in draw:
                if (xA >= mA and yA >= nA and xB <= mB and yB <=nB)or(xA <= mA and yA <= nA and xB >= mB and yB >=nB) :
                    flag=1
                    break
            if flag == 0 :
                draw.append([xA, yA, xB, yB])
            flag=0


 

        for (xA, yA, xB, yB) in draw:
            cv2.rectangle(img, (xA, yA), (xB, yB), (0,255,0), 2)
        
   # if not pick:
        
    #cv2.imshow("Raw Detections after NMS", img)
   # img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     plt.show()
    print('okkkk')
    return img

    #### Save the images below
    # k = cv2.waitKey(0) & 0xFF 
    # if k == 27:             #wait for ESC key to exit
    #     cv2.destroyAllWindows()
    # elif k == ord('s'):
    #     cv2.imwrite('Path\to_the_directory\of_saved_image.png',img)
    #     cv2.destroyAllWindows()
    
# img= cv2.imread("test1.jpg")
# process_image(img)    
   


# In[4]:


import imageio
#imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML
empty_frame_counter = 0
last_sc = np.array([])
last_rects = np.array([])
output = 'resul3.mp4'
clip = VideoFileClip("original.mp4")
video_clip = clip.fl_image(process_image)
get_ipython().run_line_magic('time', 'video_clip.write_videofile(output, audio=False)')


# In[ ]:





# In[ ]:




