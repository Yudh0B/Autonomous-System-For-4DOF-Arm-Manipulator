from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2
import mapper
import time
import serial.tools.list_ports_windows
import serial
import math
frame_id = 0
maskx=0
masky=0
no_maskx=0
no_masky=0
res_modifier=1
depth_x=0
depth_z=0
depth_y=0
x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = 0

ports=serial.tools.list_ports_windows.comports()
serialInst=serial.Serial()

portsList = []

for onePort in ports:
    portsList.append(str(onePort))
    print(str(onePort))

val = input("Select Ports : COM")

for t in range (0,len(portsList)):
    if portsList[t].startswith("COM" + str(val)):
        portVar = "COM" + str(val)
        print(portVar)
        
serialInst.baudrate = 115200
serialInst.port = portVar
serialInst.timeout = 1
serialInst.open()

net = cv2.dnn.readNet("yolov4-tiny_testing_surgical_tools_4_Classes_V9.weights", "yolov4-tiny_testing_surgical_tools_4_classes_V2.cfg")
# net = cv2.dnn.readNet("yolov7-tiny_training_surgical_tools_4_Classes_V1.weights", "yolov7-tiny_training_surgical_tools_4_classes_V1.cfg")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3)) #Generate Random Color
font = cv2.FONT_HERSHEY_SIMPLEX
timeframe = time.time()



kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                         PyKinectV2.FrameSourceTypes_Depth)
depth_width, depth_height = kinect.depth_frame_desc.Width, kinect.depth_frame_desc.Height # Default: 512, 424
color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height # Default: 1920, 1080

while True:
    
    ##############################
    ### Get images from kinect ###
    ##############################
    if kinect.has_new_color_frame() and \
       kinect.has_new_depth_frame() :
        intrinsics_matrix = kinect._mapper.GetDepthCameraIntrinsics()
        f_x = intrinsics_matrix.FocalLengthX
        f_y = intrinsics_matrix.FocalLengthY
        depth_frame = kinect.get_last_depth_frame() 

        color_img_resize = cv2.resize(cv2.cvtColor(kinect.get_last_color_frame().reshape((1080, 1920, 4)).astype(np.uint8), cv2.COLOR_RGBA2RGB), (0,0), fx=res_modifier, fy=res_modifier) # Resize (1080, 1920, 4) into half (540, 960, 4)
        depth_colormap   = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame.reshape(((depth_height, depth_width))).astype(np.uint16), alpha=255/4500), cv2.COLORMAP_JET) # Scale to display from 0 mm to 4500 mm
        

        frame_color = color_img_resize
        depth = depth_colormap
        frame_id += 1
        color_height, color_width, channels = frame_color.shape

        # Detecting Object
        blob = cv2.dnn.blobFromImage(frame_color, 1/255, (640, 640), (0, 0, 0), swapRB=False, crop=False)
        
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0 : #Confidence Level -> Accuracy
                    # Object detected
                    center_x = int(detection[0] * color_width)
                    center_y = int(detection[1] * color_height)
                    w = int(detection[2] * color_width)
                    h = int(detection[3] * color_height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (255,255,255)
                cv2.rectangle(frame_color, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame_color, label, (x, y+30), font, 1, color, 2)
                cv2.putText(frame_color, label + " " + str(round(confidence, 2)), (x, y+30), font, 1, color, 2)
                center = ((x+w/2)-(color_width/2), (y+h/2)-(color_height/2))
                real_x = 0
                real_y = 0
                pre_x = 0
                pre_y = 0
                if label == "Curved Mayo Scissor" :
                    x1 = (x+w/2)/res_modifier
                    y1 = (y+h/2)/res_modifier
                    midx=str(x1)
                    depth_x, depth_y = mapper.color_point_2_depth_point(kinect, _DepthSpacePoint, kinect._depth_frame_data, [round(x1),round(y1)]) # pixel
                    if (int(depth_y ) * 512 + int(depth_x)) < 512 * 424:
                        depth_z = float((depth_frame[int(depth_y ) * 512 + int(depth_y )])) # mm
                        if depth_z > 4500 :
                            depth_z = depth_z/3
                    else:
                        # If it exceeds return the last value to catch overflow
                        depth_z = float(depth_frame[int((512 * 424) - 1)]) # mm
                        if depth_z > 4500 :
                            depth_z = depth_z/3
                    if depth_x != 0 and depth_y != 0 and depth_z != 0 :
                        x_world = (256 - depth_x) * depth_z/f_x
                        y_world = (212 - depth_y) * depth_z/f_y
                        print("Curved Mayo Scissor", x_world, y_world, depth_z)
                    serialInst.write(bytes(b'%r' % x1))
                if label == "Mayo Needle Holder":
                    x2 = (x+w/2)/res_modifier
                    y2 = (y+h/2)/res_modifier
                    midx=str(x2)
                    depth_x, depth_y = mapper.color_point_2_depth_point(kinect, _DepthSpacePoint, kinect._depth_frame_data, [round(x2),round(y2)]) # pixel
                    if (int(depth_y ) * 512 + int(depth_x)) < 512 * 424:
                        depth_z = float((depth_frame[int(depth_y ) * 512 + int(depth_y )])) # mm
                        if depth_z > 4500 :
                            depth_z = depth_z/3 
                    else:
                        # If it exceeds return the last value to catch overflow
                        depth_z = float(depth_frame[int((512 * 424) - 1)]) # mm
                        if depth_z > 4500 :
                            depth_z = depth_z/3
                    if depth_x != 0 and depth_y != 0 and depth_z != 0 :
                        x_world = (256 - depth_x) * depth_z/f_x
                        y_world = (212 - depth_y) * depth_z/f_y
                        print("Mayo Needle Holder", x_world, y_world, depth_z)
                    serialInst.write(bytes(b'%r' % x2))
                if label == "Gillies Toothed Dissector" :
                    x3 = (x+w/2)/res_modifier
                    y3 = (y+h/2)/res_modifier
                    midx=str(x3)
                    depth_x, depth_y = mapper.color_point_2_depth_point(kinect, _DepthSpacePoint, kinect._depth_frame_data, [round(x3),round(y3)]) # pixel
                    if (int(depth_y ) * 512 + int(depth_x)) < 512 * 424:
                        depth_z = float((depth_frame[int(depth_y ) * 512 + int(depth_y )])) # mm
                        if depth_z > 4500 :
                            depth_z = depth_z/3
                    else:
                        # If it exceeds return the last value to catch overflow
                        depth_z = float(depth_frame[int((512 * 424) - 1)]) # mm
                        if depth_z > 4500 :
                            depth_z = depth_z/3
                    if depth_x != 0 and depth_y != 0 and depth_z != 0 :
                        x_world = (256 - depth_x) * depth_z/f_x
                        y_world = (212 - depth_y) * depth_z/f_y
                        print("Gillies Toothed Dissector", x_world, y_world, depth_z)
                    serialInst.write(bytes(b'%r' % x3))
                if label == "Dressing Scissor" :
                    x4 = (x+w/2)
                    y4 = (y+h/2)
                    midx=str(x4)
                    depth_x, depth_y = mapper.color_point_2_depth_point(kinect, _DepthSpacePoint, kinect._depth_frame_data, [round(x4),round(y4)]) # pixel
                    if (int(depth_y ) * 512 + int(depth_x)) < 512 * 424:
                        depth_z = float((depth_frame[int(depth_y ) * 512 + int(depth_y )])) # mm
                        if depth_z > 4500 :
                            depth_z = depth_z/3
                    else:
                        # If it exceeds return the last value to catch overflow
                        depth_z = float(depth_frame[int((512 * 424) - 1)]) # mm
                        if depth_z > 4500 :
                            depth_z = depth_z/3
                    if depth_x != 0 and depth_y != 0 and depth_z != 0 :
                        x_world = (256 - depth_x) * depth_z/f_x
                        y_world = (212 - depth_y) * depth_z/f_y
                        print("Dressing Scissor", x_world, y_world, depth_z)
                    serialInst.write(bytes(b'%r' % x4))
                    print(x4)
            cv2.circle(depth_colormap, (depth_x,depth_y), radius=5, color=(0, 0, 255), thickness=5)

        elapsed_time = time.time() - timeframe
        fps = frame_id / elapsed_time
        cv2.putText(frame_color, str(round(fps,2)), (10, 50), font, 2, (255, 255, 255), 2) #FPS Value
        cv2.putText(frame_color, "FPS", (220, 50), font, 2, (255, 255, 255), 2) #FPS Label
        cv2.putText(depth_colormap, str(round(fps,2)), (10, 50), font, 2, (255, 255, 255), 2) #FPS Value
        cv2.putText(depth_colormap, "FPS", (220, 50), font, 2, (255, 255, 255), 2) #FPS Label
        frame_color = cv2.resize(frame_color, [960,540])
        cv2.imshow('color', frame_color)
        cv2.imshow('depth', depth_colormap)

        cv2.waitKey(30)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        
kinect.close()
cv2.destroyAllWindows