<div align="center"><img src="https://github.com/nelson123-lab/Gender_based_cleaning_algorithm/blob/74da94d77c0da72f5769494e3df7320510bcbc7e/Data/All%20Gender%20(1).png" width="900"/></div>


# Gender Based Cleaning Algorithm

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/nelson123-lab/Gender_based_cleaning_algorithm/issues)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=255)](https://www.linkedin.com/in/nelsonjoseph123/)
[![Youtube](https://img.shields.io/badge/-Youtube-black.svg?style=flat-square&logo=Youtube&colorB=900)](https://www.youtube.com/channel/UCj-j1k_3vC6F1rVgrEhDF7g)
[![Medium](https://img.shields.io/badge/-Medium-black.svg?style=flat-square&logo=Medium&colorB=000)](https://medium.com/me/stories/public)


Data cleansing is the major use for this algorithm. It aids in determining the gender of an image by looking at the face. The image is erased if the face cannot be located. The algorithm can be altered to suit different requirements.

## General Script

```python

from deepface import DeepFace # Pretrained model which is present in DeepFace library.
from tqdm import tqdm # Used to create a bar that represents process progress.
import cv2
import matplotlib.pyplot as plt
import time
import os
start = time.time()
# plt.imshow(img[:,:,::-1])
# plt.show() # To display the image if required.

dire = r"Location of folder in which all the files are present"
for img in tqdm(os.listdir(dire)):
    path = dire+'/'+img
    try:
        # print(path)
        img = cv2.imread(path)
        result = DeepFace.analyze(img, actions= ['gender'])
        # print("Gender: ", result['gender']) 
        if result['gender'] # We can make changes here for custom use.
            os.remove(path)
    except ValueError:
        os.remove(path)
print("All is done.") # To understand that all the process is finished.
time.sleep(1)
end = time.time()
print(f"Runtime of the program is {end-start}") # To print out the final execution time.

```
## Use Cases:-

## 1. To eliminate noisy photos and only keep images with human faces.

<div align="center"><img src="https://github.com/nelson123-lab/Gender_based_cleaning_algorithm/blob/6ab531cc304eaa80d52a02556cf2a75abd2b9845/Data/Unprocessed%20data.png" width="900"/></div>
Multiple photographs taken from the internet are combined in the folder. The files contain photos of various genders, some of which are corrupt. These noisy photos can be removed with the help of our script. 


### Noise in Face data

<p align="center"><img src="https://github.com/nelson123-lab/Gender_based_cleaning_algorithm/blob/e44635725851404a1143618f275c41d1329ddb59/Data/Noise%20in%20face%20data.png" width="400" height="440"></p>

The noisy images displayed here are not just arbitrary snapshots. In reality, these are pictures that in some way depict the attributes of a face. These are the results of a face detector model using MTCNN that was cropped out.

### Implementation

We only need to make changes to one line of the general script as follows:-
```python
if result['gender'] != "Man" and result['gender'] != "Woman": #change the General script with this line of code.
    os.remove(path)

```
After running the script we will obtain the following results as shown below.

<div align="center"><img src="https://github.com/nelson123-lab/Gender_based_cleaning_algorithm/blob/6ab531cc304eaa80d52a02556cf2a75abd2b9845/Data/Cleaned_data_all_gender.png" width="900"/></div>

The only photographs left are those with human faces.

<div align="center"><img src="https://github.com/nelson123-lab/Gender_based_cleaning_algorithm/blob/6ab531cc304eaa80d52a02556cf2a75abd2b9845/Data/Time_taken%20and_count.png" width="900"/></div>

Progress bar is shown for understanding the cleaning status. 
Total execution time will be printed out at the end along with the text "All is done".

## 2. To determine how many photos contain human faces.

This uses the same directory as above. We must add a variable count and make the appropriate adjustments in order to determine the number of photos that contain human faces.

```python
from deepface import DeepFace
from tqdm import tqdm
import cv2
import os

dire = r"Location of folder in which all the files are present"
count = 0 #Initiated count
for img in tqdm(os.listdir(dire)):
    path = dire+'/'+img
    try:
        img = cv2.imread(path)
        result = DeepFace.analyze(img, actions= ['gender'])
        if result['gender'] == "Man" or result['gender'] == "Woman":
            count += 1 # Count value is incremented when a face is found.
    except ValueError:
        pass
print("No of human faces =",count)
```
Output is given as 

```python
No of human faces = 9
```

## 3. To only save pictures with male faces.

### Implementation

We only need to make changes to one line of the general script as follows:-
```python
if result['gender'] != "Man" #change the General script with this line of code.
    os.remove(path)
```

After executing the script, we will receive a folder with only photographs of men in it and the rest empty.

<p align="center"><img src="https://github.com/nelson123-lab/Gender_based_cleaning_algorithm/blob/6ab531cc304eaa80d52a02556cf2a75abd2b9845/Data/Only%20man.png" width="500" height="300"></p>

## 4. To only save pictures of women's faces.

### Implementation

We only need to make changes to one line of the general script as follows:-
```python
if result['gender'] != "Woman" #change the General script with this line of code.
    os.remove(path)

```
After executing the script, we will receive a folder with just photographs of women in it, with the rest of the images being deleted.

<div align="center"><img src="https://github.com/nelson123-lab/Gender_based_cleaning_algorithm/blob/6ab531cc304eaa80d52a02556cf2a75abd2b9845/Data/Only%20women.png" width="900"/></div>

# Dependency Installation

The essential libraries can be downloaded from ['PyPI'](https://pypi.org/) for installation. The libraries themselves as well as their requirements will be installed.

```python
pip install deepface
```
-Deepface is a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python. It is a hybrid face recognition framework wrapping state-of-the-art models: VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace and Dlib. The library is mainly powered by TensorFlow and Keras.
Experiments show that human beings have 97.53% accuracy on facial recognition tasks whereas those models already reached and passed that accuracy level.

```python
pip install tqdm
```
-tqdm instantly make your loops show a smart progress meter - just wrap any iterable with tqdm(iterable), and you’re done!

```python
pip install opencv-python
```
-OpenCV (Open Source Computer Vision Library: http://opencv.org) is an open-source library that includes several hundreds of computer vision algorithms.

```python
pip install matplotlib
```
-Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

-The time and OS modules are part of Python's standard library. So no need to download it.

Then you will be able to import the libraries and use its functionalities. 

## Contribution

Pull requests are welcome.
