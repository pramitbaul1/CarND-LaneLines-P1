
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[25]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().magic('matplotlib inline')


# ## Read in an Image

# In[26]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteCurve.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[27]:


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    sl_left=[]
    sl_right=[]
    ver_left=[]
    ver_right=[]
    ver=[]
    
    for line in lines:  # For every line in HoughLines
        for x1,y1,x2,y2 in line:   #For Each set of points in the line
               temp=np.array([(x1,y1),(x2,y2)])
               sl=(y2-y1)/(x2-x1)
               if sl < -0.3 :
                   sl_left.append(sl)
                   ver_left.append(line)
               elif sl > 0.3:
                   sl_right.append(sl)
                   ver_right.append(line)
    
    # Find average slope
    imshape=img.shape
    vertices = np.array([[(0,imshape[0]),(int(0.45*imshape[1]), int(0.6*imshape[0])), (int(0.6*imshape[1]), int(0.6*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)
    if len(sl_left)>0:
        sl_left_avg=sum(sl_left)/len(sl_left)
    if len(sl_right)>0:
        sl_right_avg=sum(sl_right)/len(sl_right)
    ver_left_array=np.asarray(ver_left)
    ver_right_array=np.asarray(ver_right)
    row=ver_left_array.shape[0]
    col=ver_left_array.shape[2]
    ver_left_array=ver_left_array.reshape(int(row*2),int(col/2))
    row=ver_right_array.shape[0]
    col=ver_right_array.shape[2]
    ver_right_array=ver_right_array.reshape(int(row*2),int(col/2))
    col=1
    ver_left_array=ver_left_array[np.argsort(ver_left_array[:,col])]
    ver_right_array=ver_right_array[np.argsort(ver_right_array[:,col])]
    #y=mx+c, c=y-m*x
    
    # Left Lane: find c
    xx=np.transpose(ver_left_array[:,0])
    yy=np.transpose(ver_left_array[:,1])
    left=np.polyfit(xx,yy,1)
    xx=np.transpose(ver_right_array[:,0])
    yy=np.transpose(ver_right_array[:,1])
    right=np.polyfit(xx,yy,1)
    ymin=img.shape[0]-1
    ymax=vertices[0][1][1]
    m,c=left
    xmin_left=int((ymin-c)/m)
    xmax_left=int((ymax-c)/m)
    
    m,c=right
    xmin_right=int((ymin-c)/m)
    xmax_right=int((ymax-c)/m)
    fline_left=np.array([xmin_left,ymin,xmax_left,ymax])
    fline_right=np.array([xmin_right,ymin,xmax_right,ymax])
    final_lines=np.array([fline_left,fline_right])
    final_lines=final_lines.reshape([final_lines.shape[0],1,final_lines.shape[1]])
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, final_lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
def pipeline(img):
    imGray=grayscale(img)
    imgCanny=canny(imGray,50,150)
    imgGauss=gaussian_blur(imgCanny,5)
    imshape=img.shape
    vertices = np.array([[(0,imshape[0]),(int(0.45*imshape[1]), int(0.6*imshape[0])), (int(0.6*imshape[1]), int(0.6*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)
    imgROI = region_of_interest(imgGauss,vertices)
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 80     # minimum number of votes (intersections in Hough grid cell). 
    min_line_length = 3 #minimum number of pixels making up a line
    max_line_gap = 1 
    imgHough = hough_lines(imgROI,  rho, theta, threshold, min_line_length, max_line_gap)
    final_img = weighted_img(imgHough,img,α=0.8, β=1., λ=0.)
    target=final_img
    return target
    


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[28]:


import os
List = os.listdir("test_images/")
print(List)


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[29]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
# Pipeline for images
# Step1: Convert to Grayscale
#reading in an image
for pic in List: 
    
    image = cv2.imread("test_images/"+pic)
    img=pipeline(image)
    #printing out some stats and plotting
    plt.figure()
    plt.imshow(img,cmap='gray')
    cv2.imwrite("output_images/output_"+pic,img)


# In[30]:


plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
# Step2: Get Canny Image
imgCanny=canny(img,50,175)
#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(imgCanny,cmap='gray')
# Step3: Get hough transform


# In[31]:


# Step3: Apply Gaussian Blur
imgGauss=gaussian_blur(imgCanny, 5)
plt.imshow(imgGauss,cmap='gray')


# In[32]:


# Step4: Apply an ROI
imshape=image.shape
vertices = np.array([[(0,imshape[0]),(425, 325), (575, 325), (imshape[1],imshape[0])]], dtype=np.int32)
vertices = np.array([[(0,imshape[0]),(int(0.45*imshape[1]), int(0.6*imshape[0])), (int(0.6*imshape[1]), int(0.6*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)
imgROI=region_of_interest(imgGauss, vertices)
plt.imshow(imgROI,cmap='gray')
print(imshape)
print(vertices)
print(vertices[0][3])


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[33]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[34]:


def process_image(image):
   result=pipeline(image)
   return result


# Let's try the one with the solid white lane on the right first ...



# In[35]:
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')

#Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.
# %% Video Debug
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
name=" solidWhiteRight"
count=0
e=[]
for frame in clip1.iter_frames():
    try:
        q=process_image(frame)
        cv2.imwrite('correct_images_solidWhiteRight/Frame_'+str(count)+'.jpg',cv2.cvtColor(q, cv2.COLOR_RGB2BGR))
    except IndexError:
        cv2.imwrite('error_images_solidWhiteRight/Frame_'+str(count)+'.jpg',cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        e.append(count)
    count=count+1

clip2 = VideoFileClip("test_videos/solidYellowLeft.mp4")
count=0
e1=[]
for frames in clip2.iter_frames():
    try:
        q=process_image(frames)
        cv2.imwrite('correct_images_solidYellowLeft/Frame_'+str(count)+'.jpg',cv2.cvtColor(q, cv2.COLOR_RGB2BGR))
    except IndexError:
        cv2.imwrite('error_images_solidYellowLeft/Frame_'+str(count)+'.jpg',cv2.cvtColor(frames, cv2.COLOR_RGB2BGR))
        e1.append(count)
    count=count+1
# %% Debugging each image

# In[22]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[23]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# In[36]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[37]:


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[16]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

#%%   LUCAS, Debugging Starts here
    
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 150     # minimum number of votes (intersections in Hough grid cell). 
    min_line_length = 5 #minimum number of pixels making up a line
    max_line_gap = 1 
    # lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
#%%
    image=cv2.imread('error_images/solidWhiteRight4.jpg')
    img=np.copy(image)
    imGray=grayscale(img)
    imgCanny=canny(imGray,50,150)
    imgGauss=gaussian_blur(imgCanny,5)
    imshape=img.shape
    vertices = np.array([[(0,imshape[0]),(int(0.45*imshape[1]), int(0.6*imshape[0])), (int(0.6*imshape[1]), int(0.6*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)
    imgROI = region_of_interest(imgGauss,vertices)    
    #imgHough = hough_lines(imgROI,  rho, theta, threshold, min_line_length, max_line_gap)
    lines = cv2.HoughLinesP(imgROI, rho, theta, threshold, np.array([]),min_line_length,max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#for line in lines:  # For every line in HoughLines
#    for x1,y1,x2,y2 in line:   #For Each set of points in the line
#           temp=np.array[x1,y1,x2,y2]
#           sl=(y2-y1)/(x2-x1)
#           if sl<0 :
#               sl_left=np.append(sl_left,sl)
#           elif sl>0:
#               sl_right=np.append(sl_right,sl)
#               ver_left=np.append(ver_left,np.array([x1,y1,x2,y2])

#%% Draw Line mod
sl_left=[]
sl_right=[]
ver_left=[]
ver_right=[]
ver=[]

for line in lines:  # For every line in HoughLines
    for x1,y1,x2,y2 in line:   #For Each set of points in the line
           temp=np.array([(x1,y1),(x2,y2)])
           sl=(y2-y1)/(x2-x1)
           if sl < 0 :
               sl_left.append(sl)
               ver_left.append(line)
               #ver_left=np.concatenate((ver_left,temp),axis=0)
           elif sl > 0:
               sl_right.append(sl)
               ver_right.append(line)
# Find average slope
sl_left_avg=sum(sl_left)/len(sl_left)
sl_right_avg=sum(sl_right)/len(sl_right)
ver_left_array=np.asarray(ver_left)
ver_right_array=np.asarray(ver_right)
row=ver_left_array.shape[0]
col=ver_left_array.shape[2]
print(ver_left_array.shape)
ver_left_array=ver_left_array.reshape(int(row*2),int(col/2))
row=ver_right_array.shape[0]
col=ver_right_array.shape[2]
ver_right_array=ver_right_array.reshape(int(row*2),int(col/2))
col=1
ver_left_array=ver_left_array[np.argsort(ver_left_array[:,col])]
ver_right_array=ver_right_array[np.argsort(ver_right_array[:,col])]
#y=mx+c, c=y-m*x

# Left Lane: find c
xx=np.transpose(ver_left_array[:,0])
yy=np.transpose(ver_left_array[:,1])
left=np.polyfit(xx,yy,1)
xx=np.transpose(ver_right_array[:,0])
yy=np.transpose(ver_right_array[:,1])
right=np.polyfit(xx,yy,1)
ymin=img.shape[0]-1
ymax=vertices[0][1][1]
m,c=left
xmin_left=int((ymin-c)/m)
xmax_left=int((ymax-c)/m)

m,c=right
xmin_right=int((ymin-c)/m)
xmax_right=int((ymax-c)/m)
fline_left=np.array([xmin_left,ymin,xmax_left,ymax])
fline_right=np.array([xmin_right,ymin,xmax_right,ymax])
final_lines=[fline_left,fline_right]
fl_img=np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)


# For both lines the upper and lower