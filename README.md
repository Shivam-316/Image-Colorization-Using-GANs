![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
# Image-Colorization-Using-GANs
The translation of an input image into a corresponding output image is a common problem in image processing, graphics, and vision. Even though the scenario is always the same i.e. to map pixels to pixels, these challenges are frequently solved with application-specific techniques. Conditional adversarial nets are a general-purpose approach that looks to be effective for a wide range of these issues. Automatic image colorization has piqued attention for a variety of applications, including the restoration of aged or deteriorated photos.

### Dataset

------------

This Dataset contains 7129 colourful RGB images & 7129 grayscale images of landscapes in jpg image format. Images consists of streets, buildings, mountains, glaciers, trees etc and their corresponding grayscale image in two different folders.

[Landscape color and grayscale images Dataset](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization "Landscape color and grayscale images Dataset")

### Processing Images

------------
Pre-Processing include common image related function such as **loading**, **resizing**, **random cropping plus rotation** etc. And some image color-space transformation function like **rgb2lab** and vice versa.

**RGB -> LAB**
```python
@tf.function()
def rbg2lab(target_img, isnormalized = True, normalize_lab = False, comp = ''):
  if not isnormalized:
    target_img = target_img/255.0                                               # normalizes the rgb image in range 0-1
  
  # Takes RGB Image in Normalized Form
  target_img = tfio.experimental.color.rgb_to_lab(target_img)
  if normalize_lab:
    tf.Assert(tf.reduce_any(tf.equal(comp, ['vis', 'net'])), data=[comp], name='Lab_Normalization_Error')
    if comp == 'vis':
      target_img = (target_img + [0, 128, 128]) / [100., 255., 255.]            # normalizes in 0-1 range for visualization of image
    else:
      target_img = target_img / [50., 127.5, 127.5] + [-1, 0., 0.]              # normalizes in -1 to 1 range for neural networks as they perform better in this range
  return target_img
```
**LAB -> RGB**
```python
@tf.function()
def lab2rgb(lab_img, isnormalized = False, comp = ''):
  if isnormalized:
    tf.Assert(tf.reduce_any(tf.equal(comp, ['vis', 'net'])), data=[comp], name='Lab_Normalization_Error')
    if comp == 'vis':
      lab_img = lab_img * [100.,255., 255.] + [0, -128, -128];                  # from 0-1 range
    else:
      lab_img = (lab_img + [1.,0., 0.]) * [50., 127.5, 127.5];                  # from -1 to 1 range
  
  # Takes LAB Image in Unnormalized Form.
  rgb = tfio.experimental.color.lab_to_rgb(lab_img) 
  return rgb
```
### Model Architecture

------------

#### Generator
The input (*grayscale image*) is transmitted through a number of layers that gradually downsample the data until it reaches a bottleneck layer, at which time the process is reversed and the latent representation is upsampled into an colored output image, we add skip connections in the shape of a **"U-Net"**. We add skip connections between each layer *i* and layer *n-i* where *n* is the total number of layers, in particular. Each skip connection simply concatenates all layers *i* and *n-i* channels.

<img src = 'https://user-images.githubusercontent.com/56474719/167465640-68d334a8-9f90-4a2e-8a01-8727938a1aba.png' width="400" height="500">


#### Discriminator
The discriminator attempts to determine whether each of the *N x N* patches in a picture is authentic or phoney. The final output of *D* is calculated by averaging all responses.

<img src = 'https://user-images.githubusercontent.com/56474719/167465830-57546a98-a34b-42bc-bba9-1dcc047b0ef4.png' width="300" height="400">


#### Custom GAN Model
This model combines both the generator and discriminator, calculates losses and matrices, performs gradient descent steps and displays results.

Gist Link : [GAN Architecture](https://gist.github.com/Shivam-316/07593d677dbceaff630a5928e7e22d95 "GAN Architecture")

### Results

------------

![img127](https://user-images.githubusercontent.com/56474719/167465982-261c0c84-8d77-4964-8dd1-6655b6fa5174.jpg)
![img133](https://user-images.githubusercontent.com/56474719/167465998-37b726ea-265b-4cc3-9c2f-c2e1c8e7ecd9.jpg)


![img149](https://user-images.githubusercontent.com/56474719/167466011-97c18637-c838-4655-a579-4bee9d031883.jpg)
![img137](https://user-images.githubusercontent.com/56474719/167466066-721ca87f-1b81-4397-a87b-1279a75b8b87.jpg)

