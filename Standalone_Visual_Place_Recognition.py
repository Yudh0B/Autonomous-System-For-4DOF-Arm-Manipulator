from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import numpy as np
import math
import cv2
import numpy as np
import imutils
from keras import models
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array

res_modifierx=(1920/224)
res_modifiery=(1080/224)
res_modifier = 0.5

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height # Default: 1920, 1080

model = models.load_model(
    'C:\TA\Jalu\VPR_VGG16_ES.h5')

font = cv2.FONT_HERSHEY_SIMPLEX
count_classroom = 0
count_corridor = 0
count_labelka = 0
count_labpower = 0

while True:
    
    ##############################
    ### Get images from kinect ###
    ##############################
    if kinect.has_new_color_frame():

        color_img_resize = cv2.resize(cv2.cvtColor(kinect.get_last_color_frame().reshape((1080, 1920, 4)).astype(np.uint8), cv2.COLOR_RGBA2RGB), (0,0), fx=res_modifier, fy=res_modifier) # Resize (1080, 1920, 4) into half (540, 960, 4)
        color_tensor = cv2.resize(color_img_resize, (224, 224))

        frame_color = color_tensor
        color_height, color_width, channels = frame_color.shape


        frame_array = img_to_array(frame_color)
        frame_ED = np.expand_dims(frame_array, axis=0)
        frame_proccessed = imagenet_utils.preprocess_input(frame_ED)

        prediction = np.argmax(model.predict(frame_proccessed), axis=1)
        print(prediction)
        
        if prediction == 0:
            print("classroom")
            count_classroom = count_classroom+1
        elif prediction == 1:
            print("corridor")
            count_corridor = count_corridor+1
        elif prediction == 2:
            print("labelka")
            count_labelka = count_labelka+1
        elif prediction == 3:
            print("labpower")
            count_labpower = count_labpower+1
        else:
            print("Invalid")

        if count_classroom == 20:
            print("this is a classroom")
            break
        elif count_corridor == 20:
            print("this is a corridor")
            break
        elif count_labelka == 20:
            print("this is labelka")
            break
        elif count_labpower == 20:
            print("this is labpower")
            break

        cv2.imshow('Prediction', color_img_resize)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
frame_color.release()
cv2.destroyAllWindows()