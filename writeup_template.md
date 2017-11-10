# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./examples/canny.jpg "Canny"
[image3]: ./examples/gauss.jpg "Gauss"
[image4]: ./examples/mask.jpg "Mask"
[image5]: ./examples/line-segments-example.jpg "Lane Markings"
[image6]: ./examples/final.jpg "Extrapolated Road Lines"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, 
![alt text][image1]

I then ran a canny edge detector on the grayscale image to detect the edges on the image. 
![alt text][image2]

I applied a gaussian filter to make the detected edges more prominent by sharpening their features. 
![alt text][image3]

I created a mask to identify the region of interest, which in this case is the lane occupied by the vehicle. The mask is static, but needs to have a moving horizon if the road has 
a gradient. 
![alt text][image4]

I used the given hough transform and draw_lines function to annotate lane markings, similar to the example image provided in the repository 
![alt text][image5]

I modified the hough_lines function by separating the left and right lines with a threshold parameter. (A slope of -0.3  for the left lane and 0.3 for the right lane). I found the average
of  the left and right lane slopes and fitted a first order polynomial on both sides. I used these lines as input to the draw_lines which is unchanged. An example of an image after processed 
by my pipeline can be seen below. 
![alt text][image6]

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when no line is detected. Some of the mathematical operations in draw line assume that every frame detects a line. 


Another shortcoming could be using it for curved/winding roads. The hough transform votes for straight lines and the program essentially tries to fit in straight line equations for all 
cases


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to implement a moving average alogorithm into the draw_lines function so that the output video is smoother in terms of line detection

Another potential improvement could be to add error handling for IndexError and DivisionByZero error to better handle edge cases. Fitting 2nd or 3rd order polynomials to road lines 
would better represent real world conditions.  
