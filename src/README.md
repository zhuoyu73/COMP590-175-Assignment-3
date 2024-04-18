# Instruction on How to Use Problem1.py file

This Problem1.py file is for COMP590-175 Assignment 3 Problem 1. It uses skimage package to process images. 

Python Initials: Line 13-27
1. Use skimage function imread for reading .tiff images and .nef (raw) images.
2. Print image width, height, and bits per pixel.
3. Convert the image into a double-precision array

Linearization: Line 30-36
1. Apply a linear transformation to the double-precision image to obtain a normalized image

Identify the correct Bayer pattern: Line 39-45
1. Extract the top left 2x2 squares of the .tiff image file and the normalized image file
2. Identify the Bayer pattern based on the result

White Balancing Using .nef image (three channels): Line 48-81
1. Obtain the multiplier scale information from dcraw
2. Conduct white balance with white world, grey world, and camera preset algorithms with the .nef image (three channels)

White Balancing Using .tiff image (one channel): Line 84-155
1. Obtain the three channel information based on the identification of the Bayer pattern. Here, we identified "rggb" as our Bayer pattern, so we extracted the three channels and calculated the average and the scaling factors.
2. Apply while balancing using white world, grey world, and camera preset algorithms with different factors and scaling information

Demosaicing: Line 158-172
1. Based on the white balancing images, here we chose camera preset white balancing result for remaining processing.
2. Here we use interp2d to create interpolation functions for each color channel
3. Interpolate each color channel and stack the interpolate channels to form the image

Color Correction: Line 176-197
1. Obtain sRGB standard information
2. Obtain camera information from dcraw.c file. Here, our image is by Nikon D3.
3. Calculate the transformation matrix from Camera to sRGB
4. Invert the transformation matrix and apply the color space transformation

Brightness Adjustment & Gamma Encoding: Line 203-224
1. After the image is corrected, we conduct linear scaling to brighten the image. We first calculate the scale factor with a mean target value set to 0.25
2. Apply brightness adjustment by multiplying the original image by the scale factor
3. Gamma encodes the Image using a nonlinear operation that adjusts the luminance to match the non-linear response of the human eye better

Compression: Line 227-230
1. Compress the image using imsave function
2. Save the images in .png and .jpg formats. The quality ratio here is set as 95, but can be modified.

Problem 1.2-Manual White Balancing: Line 233-277
1. 

Problem 1.2-Manual White Balancing using processed (gamma-encoded) image: Line 280-302
1. 

