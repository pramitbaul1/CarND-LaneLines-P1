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
[image4]: ./examples/gauss.jpg "Gauss"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, 
![alt text][image1]

I then ran a canny edge detector on the grayscale image to detect the edges on the image. 
![alt text][image2]

I applied a gaussian filter to make the detected edges more prominent by sharpening their features. 
![alt text][image3]
If you'd like to include images to show how the pipeline works, here is how to include an image: 

I created a mask to identify the region of interest, which in this case is the lane occupied by the vehicle. The mask is static, but needs to have a moving horizon if the road has 
a gradient. 

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when no line is detected. Some of the mathematical operations in draw line assume that every frame detects a line. 


Another shortcoming could be using it for curved/winding roads. The hough transform votes for straight lines and the program essentially tries to fit in straight line equations for all 
cases


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to implement a moving average alogorithm into the draw_lines function so that the output video is smoother in terms of line detection

Another potential improvement could be to add error handling for IndexError and DivisionByZero error to better handle edge cases. Fitting 2nd or 3rd order polynomials to road lines 
would better represent real world conditions.  
