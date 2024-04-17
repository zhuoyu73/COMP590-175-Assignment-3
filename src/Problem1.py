#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
#COMP590-175 Assignment 3

from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.io import imsave
from skimage import img_as_ubyte

#define image
image = imread('/Users/macbookpro/Desktop/ISP_homework/data/baby.tiff')
raw = imread('/Users/macbookpro/Desktop/ISP_homework/data/baby.nef')

#Python Initials
print("Image width: ", image.shape[1])
print("Image height: ", image.shape[0])
print("Data type: ", image.dtype)

if image.dtype == 'uint16':
    print("Bits per pixel: 16")
elif image.dtype == 'uint8':
    print("Bits per pixel: 8")

image_double = image.astype('float64')


###Linearization###
black = 0
white = 16383
normalized_image = (image_double - black) / (white - black)
normalized_image = np.clip(normalized_image, 0, 1)
print("minimum:", normalized_image.min())
print("maximum:", normalized_image.max())


###Identifying the correct Bayer pattern###
top_left_2x2 = image[0:2, 0:2]
top_left_2x2_test = normalized_image[0:2, 0:2]
print("Top-left 2x2 pixel values:")
print(top_left_2x2)
print("Top-left 2x2 pixel values after linear transformation:")
print(top_left_2x2_test)


###White Balancing using nef image###
#scales from dcraw
r_scale = 1.628906
g_scale = 1.000000
b_scale = 1.386719
scales = np.array([r_scale, g_scale, b_scale])[None, None, :]
#white world
white_world_balanced_image = raw / raw.max(axis=(0, 1))
#gray world
gray_world_balanced_image = raw / raw.mean(axis=(0, 1))
#camera preset
if raw.dtype == np.uint8:
    raw = raw.astype('float64') / 255
camera_preset_balanced_image = raw * scales
camera_preset_balanced_image = np.clip(camera_preset_balanced_image, 0, 1)
#process
white_world_balanced_image = np.clip(white_world_balanced_image, 0, 1)
gray_world_balanced_image = np.clip(gray_world_balanced_image, 0, 1)
camera_preset_balanced_image = np.clip(camera_preset_balanced_image, 0, 1)
#White world results
plt.imshow(white_world_balanced_image)
plt.title('White World White Balance')
plt.show()
plt.close()
#Gray world results
plt.imshow(gray_world_balanced_image)
plt.title('Gray World White Balance')
plt.show()
plt.close()
#Camera preset results
plt.imshow(camera_preset_balanced_image)
plt.title('Camera Preset White Balance')
plt.show()
plt.close()


###White Balancing using tiff image###
#white world 
w_red_channel = normalized_image[::2, ::2]
w_green_channel = normalized_image[::2, 1::2]
w_green_channel_2 = normalized_image[1::2, ::2]
w_blue_channel = normalized_image[1::2, 1::2]
#calculate the average for each channel
w_red_avg = np.mean(w_red_channel)
w_green_avg = np.mean((w_green_channel + w_green_channel_2) / 2)
w_blue_avg = np.mean(w_blue_channel)
#calculate the scaling factors for each channel
w_red_scale = w_green_avg / w_red_avg
w_blue_scale = w_green_avg / w_blue_avg
#apply white balancing by scaling each channel
w_balanced_red_channel = w_red_channel * w_red_scale
w_balanced_green_channel = (w_green_channel + w_green_channel_2) / 2
w_balanced_blue_channel = w_blue_channel * w_blue_scale
#clip the values
w_balanced_red_channel = np.clip(w_balanced_red_channel, 0, 1)
w_balanced_green_channel = np.clip(w_balanced_green_channel, 0, 1)
w_balanced_blue_channel = np.clip(w_balanced_blue_channel, 0, 1)
#combine the balanced channels into a single image
w_balanced_image = np.stack([w_balanced_red_channel, w_balanced_green_channel, w_balanced_blue_channel], axis=-1)
plt.imshow(w_balanced_image)
plt.title('White World White Balance')
plt.show()
plt.close()

#grey world
g_red_channel = normalized_image[::2, ::2]
g_green_channel = normalized_image[::2, 1::2]
g_green_channel_2 = normalized_image[1::2, ::2]
g_blue_channel = normalized_image[1::2, 1::2]
#calculate the average value for each channel
g_red_avg = np.mean(g_red_channel)
g_green_avg = np.mean((g_green_channel + g_green_channel_2) / 2)
g_blue_avg = np.mean(g_blue_channel)
#calculate the grey world scaling factor
grey_world_scale = 1 / np.mean([g_red_avg, g_green_avg, g_blue_avg])
#apply white balancing by scaling each channel
g_balanced_red_channel = g_red_channel * grey_world_scale
g_balanced_green_channel = (g_green_channel + g_green_channel_2) / 2 * grey_world_scale
g_balanced_blue_channel = g_blue_channel * grey_world_scale
#clip the values to ensure they are within the valid range
g_balanced_red_channel = np.clip(g_balanced_red_channel, 0, 1)
g_balanced_green_channel = np.clip(g_balanced_green_channel, 0, 1)
g_balanced_blue_channel = np.clip(g_balanced_blue_channel, 0, 1)
#combine the balanced channels into a single image
g_balanced_image = np.stack([g_balanced_red_channel, g_balanced_green_channel, g_balanced_blue_channel], axis=-1)
plt.imshow(g_balanced_image)
plt.title('Gray World White Balance')
plt.show()
plt.close()

#camera presets
c_red_channel = normalized_image[::2, ::2]
c_green_channel = normalized_image[::2, 1::2]
c_green_channel_2 = normalized_image[1::2, ::2]
c_blue_channel = normalized_image[1::2, 1::2]
c_balanced_red_channel = c_red_channel * r_scale
c_balanced_green_channel = (c_green_channel + c_green_channel_2) / 2
c_balanced_blue_channel = c_blue_channel * b_scale
#clip
c_balanced_red_channel = np.clip(c_balanced_red_channel, 0, 1)
c_balanced_green_channel = np.clip(c_balanced_green_channel, 0, 1)
c_balanced_blue_channel = np.clip(c_balanced_blue_channel, 0, 1)
#combine
c_balanced_image = np.stack([c_balanced_red_channel, c_balanced_green_channel, c_balanced_blue_channel], axis=-1)
plt.imshow(c_balanced_image)
plt.title('Camera Presets White Balance')
plt.show()
plt.close()


###Demosaicing###
x = np.arange(c_balanced_image.shape[1])
y = np.arange(c_balanced_image.shape[0])
f_red = interp2d(x, y, c_balanced_image[:, :, 0], kind='linear')
f_green = interp2d(x, y, c_balanced_image[:, :, 1], kind='linear')
f_blue = interp2d(x, y, c_balanced_image[:, :, 2], kind='linear')
new_x = np.linspace(0, c_balanced_image.shape[1] - 1, c_balanced_image.shape[1] * 2)
new_y = np.linspace(0, c_balanced_image.shape[0] - 1, c_balanced_image.shape[0] * 2)
demo_red = f_red(new_x, new_y)
demo_green = f_green(new_x, new_y)
demo_blue = f_blue(new_x, new_y)
demosaiced_image = np.stack([demo_red, demo_green, demo_blue], axis=-1)
plt.imshow(demosaiced_image)
plt.title('Demosaicing')
plt.show()



###Color Correction###
M_sRGB_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
#from dcraw.c file
M_XYZ_cam = np.array([
    [ 8828 / 10000, -2406 / 10000, -694 / 10000],
    [-4874 / 10000, 12603 / 10000, 2541 / 10000],
    [ -660 / 10000,  1509 / 10000, 7587 / 10000]
])
#Calculation
M_sRGB_cam = np.dot(M_XYZ_cam, M_sRGB_XYZ)
M_sRGB_cam /= np.tile(np.sum(M_sRGB_cam, axis=1)[:, None], (1, 3)) 
Cam2sRGB = np.linalg.inv(M_sRGB_cam)
for i in range(demosaiced_image.shape[0]):
    for j in range(demosaiced_image.shape[1]):
        demosaiced_image[i][j] =  np.dot(Cam2sRGB, demosaiced_image[i][j])
corrected_image = np.clip(demosaiced_image, 0, 1)
#plot
plt.imshow(corrected_image)
plt.title('Color Space Corrected Image')
plt.show()





###Brightness adjustment and gamma encoding###
rgb_image_float = img_as_float(corrected_image)
mean_target_value = 0.25
scale_factor = mean_target_value / rgb2gray(rgb_image_float).mean()
brightened_image = rgb_image_float * scale_factor

brightened_image = np.clip(brightened_image, 0, 1)

def gamma_encode(channel):
    return np.where(channel <= 0.0031308,
                    12.92 * channel,
                    1.055 * (channel ** (1 / 2.4)) - 0.055)

gamma_encoded_image = np.zeros_like(brightened_image)
for i in range(3): 
    gamma_encoded_image[..., i] = gamma_encode(brightened_image[..., i])

gamma_encoded_image = np.clip(gamma_encoded_image, 0, 1)

plt.imshow(gamma_encoded_image)
plt.title('Gamma Encoded Image')
plt.show()


###compression###
gamma_encoded_image_uint8 = img_as_ubyte(gamma_encoded_image)
imsave('/Users/macbookpro/Desktop/ISP_homework/data/gamma_image.png', gamma_encoded_image_uint8)
imsave('/Users/macbookpro/Desktop/ISP_homework/data/gamma_image.jpg', gamma_encoded_image_uint8, quality=95)


##########Problem 1.2########
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

second_image = normalized_image
print(second_image.shape)

plt.imshow(second_image)
plt.title('The white patch')
white_patch_coords = np.array(plt.ginput(n=1, timeout=0, show_clicks=True))
plt.show()
#Calculate average
man_x, man_y = int(white_patch_coords[0][0]), int(white_patch_coords[0][1])
manual_red_channel = second_image[::2, ::2]
manual_green_channel = second_image[::2, 1::2]
manual_green_channel_2 = second_image[1::2, ::2]
manual_blue_channel = second_image[1::2, 1::2]
manual_balanced_red_channel = manual_red_channel 
manual_balanced_green_channel = (manual_green_channel + manual_green_channel_2) / 2 
manual_balanced_blue_channel = manual_blue_channel 
#combine
manual_balanced_image = np.stack([manual_balanced_red_channel, manual_balanced_green_channel, manual_balanced_blue_channel], axis=-1)
# x and y
man_x_rgb = man_y // 2
man_y_rgb = man_x // 2
man_x_rgb = max(min(man_x_rgb, manual_balanced_image.shape[1] - 1), 0)
man_y_rgb = max(min(man_y_rgb, manual_balanced_image.shape[0] - 1), 0)
man_white_patch_rgb = manual_balanced_image[man_y_rgb, man_x_rgb, :]
man_rscale = man_white_patch_rgb[0] / man_white_patch_rgb[1]
man_bscale = man_white_patch_rgb[2] / man_white_patch_rgb[1]

man_balanced_red_channel = manual_red_channel * man_rscale
man_balanced_green_channel = (manual_green_channel + manual_green_channel_2) / 2 
man_balanced_blue_channel = manual_blue_channel * man_bscale

man_balanced_red_channel = np.clip(man_balanced_red_channel, 0, 1)
man_balanced_green_channel = np.clip(man_balanced_green_channel, 0, 1)
man_balanced_blue_channel = np.clip(man_balanced_blue_channel, 0, 1)
#stack
man_balanced_image = np.stack([man_balanced_red_channel, man_balanced_green_channel, man_balanced_blue_channel], axis=-1)
#plot
plt.imshow(man_balanced_image)
plt.title('Manually White Balanced Image')
plt.show()


##########Problem 1.2 using gamma-encoded image########
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

second_image = gamma_encoded_image
print(second_image.shape)

plt.imshow(second_image)
plt.title('The white patch')
white_patch_coords = np.array(plt.ginput(n=1, timeout=0, show_clicks=True))
plt.show()
#Calculate average
x, y = int(white_patch_coords[0][0]), int(white_patch_coords[0][1])
white_patch_rgb = second_image[y, x, :]
normalized_weights = white_patch_rgb / white_patch_rgb.max()
#Apply
white_balanced_image = second_image / normalized_weights
white_balanced_image = np.clip(white_balanced_image, 0, 1)
#Plot
plt.imshow(white_balanced_image)
plt.title('White Balanced Image')
plt.show()
