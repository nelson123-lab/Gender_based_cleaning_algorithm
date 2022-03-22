<div align="center"><img src="https://github.com/nelson123-lab/Gender_based_cleaning_algorithm/blob/61a640fbe70fab444ec7b21b0fa861957aeaf894/All%20Gender%20(1).png" width="900"/></div>


# Gender_based_cleaning_algorithm

This algorithm can be mainly used for cleaning data. It helps in predicting the gender of a given image. If no face is found it will be shown as no face data availabe. We are using the above features to clean the data so that we could obtain
1. Only the images that contain Human face.
2. Only the images with Faces of Males.
3. Only the images with faces of Females.


```python

from deepface import DeepFace
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import time
import os
start = time.time()
# plt.imshow(img[:,:,::-1])
# plt.show()

dire = r"Location of folder in which all the files are present"
for img in tqdm(os.listdir(dire)):
    path = dire+'/'+img
    try:
        # print(path)
        img = cv2.imread(path)
        result = DeepFace.analyze(img, actions= ['gender'])
        # print("Gender: ", result['gender'])
        if result['gender'] != "Man":
            os.remove(path)
    except ValueError:
        os.remove(path)
print("All is done.")
time.sleep(1)
end = time.time()
print(f"Runtime of the program is {end-start}")

```
