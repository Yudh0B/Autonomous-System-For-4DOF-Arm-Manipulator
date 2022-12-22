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
import imutils
from keras import models
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array 

frame_id = 0
maskx=0
masky=0
no_maskx=0
no_masky=0
res_modifier=0.5
depth_x=0
depth_z=0
depth_y=0
x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = 0

model = models.load_model(
    'C:\TA\Jalu\VPR_VGG16_ES.h5')

count_classroom = 0
count_corridor = 0
count_labelka = 0
count_labpower = 0

net = cv2.dnn.readNet("yolov4-tiny_testing_surgical_tools_4_Classes_V9.weights", "yolov4-tiny_testing_surgical_tools_4_classes_V2.cfg")
#net = cv2.dnn.readNet("yolov4-tiny_training_last.weights", "yolov4-tiny_testing.cfg")
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

print("Ruangan Yang Tersedia :")
print("0. classrom\n1. corrior\n2. labelka\n3. labpower")
room = int(input("Pilih Ruangan dengan mengetikkan angka sebelum nama ruangan :"))
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
        
        if prediction == room:
            if prediction == 0:
                print("classroom")
                count_classroom = count_classroom+1
                if count_classroom == 25:
                    print("this is a classroom")
                    break
            if prediction == 1:
                print("corridor")
                count_corridor = count_corridor+1
                if count_corridor == 25:
                    print("this is a corridor")
                    break
            if prediction == 2:
                print("labelka")
                count_labelka = count_labelka+1
                if count_labelka == 25:
                    print("this is labelka")
                    break
            if prediction == 3:
                print("labpower")
                count_labpower = count_labpower+1
                if count_labpower == 25:
                    print("this is labpower")
                    break
        else:
            print("Invalid")

        cv2.imshow('Prediction', color_img_resize)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
# frame_color.release()
cv2.destroyAllWindows()
print ("Daftar Objek Yang Bisa DIambil :")
print ("Curved Mayo Scissor\nGillies Toothed Dissector\nMayo Needle Holder\nDressing Scissor")
object=str(input("Masukan Objek Yang Ingin diambil : "))
place_x=float(input("Masukkan Koordinat X untuk meletakkan benda (dalam cm) : "))
place_y=float(input("Masukkan Koordinat y untuk meletakkan benda (dalam cm) : "))
place_z=float(input("Masukkan Koordinat z untuk meletakkan benda (dalam cm) : "))
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
                if label == object :
                    x3 = (x+w/2)/res_modifier
                    y3 = (y+h/2)/res_modifier
                    midx=str(x3)
                    depth_x, depth_y = mapper.color_point_2_depth_point(kinect, _DepthSpacePoint, kinect._depth_frame_data, [round(x3),round(y3)]) # pixel
                    if (int(depth_y ) * 512 + int(depth_x)) < 512 * 424:
                        depth_z = float((depth_frame[int(depth_y ) * 512 + int(depth_y )])) # mm
                        if depth_z > 2000 :
                            depth_z = depth_z/6
                    else:
                        # If it exceeds return the last value to catch overflow
                        depth_z = float(depth_frame[int((512 * 424) - 1)]) # mm
                        if depth_z > 2000 :
                            depth_z = depth_z/6
                    print(depth_x, depth_y, depth_z)
            cv2.circle(depth_colormap, (depth_x,depth_y), radius=5, color=(0, 0, 255), thickness=5)
            x_world = (256 - depth_x) * depth_z/f_x
            y_world = (212 - depth_y) * depth_z/f_y
            
            if x_world != 0 and y_world !=0 and depth_z <=700 :
                import os
                from numpy import *
                from math import *
                import numpy as np

                st_x = (x_world+28)*0.1
                st_y = (depth_z+155)*0.1
                st_z = (y_world+10)*0.1


                st_y = st_y*-1
                r1 = 29.92
                r2 = 29.67
                r3 = 14.1
                R = r1 + r2 + r3

                def constrain(amt,low,high):
                    if amt < low:
                        amt = low
                    elif amt > high:
                        amt = high
                    return amt

                def to_RAD(deg):
                    return float(deg)*(math.pi / 180)

                def end_point(d1, d2, d4, d6):
                    x_0 = 0
                    y_0 = 0
                    z_0 = 0

                    x_1 = r1 * math.cos(to_RAD(d1)) * math.cos(to_RAD(d2)) + x_0
                    y_1 = r1 * math.sin(to_RAD(d1)) * math.cos(to_RAD(d2)) + y_0
                    z_1 = r1 * math.sin(to_RAD(d2)) + z_0

                    r_x_1 = np.array([x_0, x_1])
                    r_y_1 = np.array([y_0, y_1])
                    r_z_1 = np.array([x_0, z_1])
                    r_1 = np.array([r_x_1, r_y_1, r_z_1])

                    x_2 = (r2 * math.cos(to_RAD(d1)) * math.cos(to_RAD(d2 + d4))) + x_1
                    y_2 = (r2 * math.sin(to_RAD(d1)) * math.cos(to_RAD(d4 + d2))) + y_1
                    z_2 = (r2 * math.sin(to_RAD(d4 + d2))) + z_1

                    r_x_2 = np.array([x_1, x_2])
                    r_y_2 = np.array([y_1, y_2])
                    r_z_2 = np.array([z_1, z_2])

                    x_3 = (r3 * math.cos(to_RAD(d1)) * math.cos(to_RAD(d2 + d4 + d6))) + x_2
                    y_3 = (r3 * math.sin(to_RAD(d1)) * math.cos(to_RAD(d4 + d2 + d6))) + y_2
                    z_3 = (r3 * math.sin(to_RAD(d4 + d2 + d6))) + z_2

                    r_x_3 = np.array([x_2, x_3])
                    r_y_3 = np.array([y_2, y_3])
                    r_z_3 = np.array([z_2, z_3])
                    return x_3, y_3, z_3

                def forward_kinematics(d1, d2, d4, d6):
                    x_0 = 0
                    y_0 = 0
                    z_0 = 0

                    x_1 = r1 * math.cos(to_RAD(d1)) * math.cos(to_RAD(d2)) + x_0
                    y_1 = r1 * math.sin(to_RAD(d1)) * math.cos(to_RAD(d2)) + y_0
                    z_1 = r1 * math.sin(to_RAD(d2)) + z_0

                    r_x_1 = np.array([x_0, x_1])
                    r_y_1 = np.array([y_0, y_1])
                    r_z_1 = np.array([x_0, z_1])
                    r_1 = np.array([r_x_1, r_y_1, r_z_1])

                    x_2 = (r2 * math.cos(to_RAD(d1)) * math.cos(to_RAD(d2 + d4))) + x_1
                    y_2 = (r2 * math.sin(to_RAD(d1)) * math.cos(to_RAD(d4 + d2))) + y_1
                    z_2 = (r2 * math.sin(to_RAD(d4 + d2))) + z_1

                    r_x_2 = np.array([x_1, x_2])
                    r_y_2 = np.array([y_1, y_2])
                    r_z_2 = np.array([z_1, z_2])

                    x_3 = (r3 * math.cos(to_RAD(d1)) * math.cos(to_RAD(d2 + d4 + d6))) + x_2
                    y_3 = (r3 * math.sin(to_RAD(d1)) * math.cos(to_RAD(d4 + d2 + d6))) + y_2
                    z_3 = (r3 * math.sin(to_RAD(d4 + d2 + d6))) + z_2

                    r_x_3 = np.array([x_2, x_3])
                    r_y_3 = np.array([y_2, y_3])
                    r_z_3 = np.array([z_2, z_3])
                    return x_3, y_3, z_3, r_x_1, r_y_1, r_z_1, r_x_2, r_y_2, r_z_2, r_x_3, r_y_3, r_z_3

                def inverse_kinematics(x, y, z):
                    a1 = math.atan2(y, x)
                    r = math.sqrt((x * x) + (y * y))
                    d = math.sqrt(((r - r3) * (r - r3)) + (z * z))
                    theta = math.atan2(z, r - r3)
                    A = ((r1 * r1) + (d * d) - (r2 * r2)) / (2 * r1 * d)
                    A = constrain(A, -1, 1)
                    A = math.acos(A)
                    a2 = theta + A

                    B = ((r1 * r1) + (r2 * r2) - (d * d)) / (2 * r1 * r2)
                    B = constrain(B, -1, 1)
                    B = math.acos(B)
                    a3 = B - math.pi
                    C = ((r2 * r2) + (d * d) - (r1 * r1)) / (2 * r2 * d)
                    C = constrain(C, -1, 1)
                    C = math.acos(C)
                    a4 = C - theta
                    #print(math.degrees(a1), math.degrees(a2), math.degrees(a3), math.degrees(a4))
                    return math.degrees(a1), math.degrees(a2), math.degrees(a3), math.degrees(a4)

                a11, a22, a33, a44 = inverse_kinematics(st_x, st_y, st_z)
                b11, b22, b33, b44 = inverse_kinematics(place_x, place_y, place_z)
                x_3, y_3, z_3, r_x_1, r_y_1, r_z_1, r_x_2, r_y_2, r_z_2, r_x_3, r_y_3, r_z_3 = forward_kinematics(a11, a22, a33, a44)
                a11 = a11*-1
                a22=a22+92
                a33=(-a33)+169.1
                a44=a44+200
                b22=b22+92
                b33=(-b33)+169.1
                b44=b44+200
                print("B11 = " + str(b11))

                if os.name == 'nt':
                    import msvcrt
                    def getch():
                        return msvcrt.getch().decode()
                else:
                    import sys, tty, termios
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    def getch():
                        try:
                            tty.setraw(sys.stdin.fileno())
                            ch = sys.stdin.read(1)
                        finally:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        return ch

                from dynamixel_sdk import *     

                # Control table address
                ADDR_MX_TORQUE_ENABLE      = 64               # Control table address is different in Dynamixel model
                ADDR_MX_GOAL_POSITION      = 116
                ADDR_MX_PRESENT_POSITION   = 132
                ADDR_MX_PROFILE_VELOCITY   = 112
                ADDR_MX_PROFILE_ACCELERATION = 108
                ADDR_D_GAIN                = 80
                ADDR_AX_TORQUE_ENABLE      = 24               # Control table address is different in Dynamixel model
                ADDR_AX_GOAL_POSITION      = 30
                ADDR_AX_PRESENT_POSITION   = 36
                ADDR_AX_MOVING_SPEED       = 32                # Speed Address
                ADDR_AX_MOVING             = 46                # Moving Status Address
                ADDR_XM_TORQUE_ENABLE      = 64               # Control table address is different in Dynamixel model
                ADDR_XM_GOAL_POSITION      = 116
                ADDR_XM_PRESENT_POSITION   = 132
                ADDR_XM_PROFILE_VELOCITY   = 112
                ADDR_XM_PROFILE_ACCELERATION = 108
                ADDR_XH_TORQUE_ENABLE      = 64               # Control table address is different in Dynamixel model
                ADDR_XH_GOAL_POSITION      = 116
                ADDR_XH_PRESENT_POSITION   = 132
                ADDR_XH_PROFILE_VELOCITY   = 112
                ADDR_XH_PROFILE_ACCELERATION = 108

                # Protocol version
                PROTOCOL_VERSION1           = 1.0               # See which protocol version is used in the Dynamixel
                PROTOCOL_VERSION2           = 2.0               # See which protocol version is used in the Dynamixel

                # Default setting
                DXL1_ID                     = 1
                DXL2_ID                     = 2
                DXL3_ID                     = 3                 # Dynamixel ID : 1
                DXL4_ID                     = 4
                DXL5_ID                     = 5
                BAUDRATE                    = 1000000            # Dynamixel default baudrate : 57600
                DEVICENAME                  = 'COM3'    # Check which port is being used on your controller
                                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
                TORQUE_ENABLE                 = 1                 # Value for enabling the torque
                TORQUE_DISABLE                = 0
                DXL_MINIMUM_POSITION_VALUE_1  = round(90/0.088)           # Dynamixel will rotate between this value
                DXL_MAXIMUM_POSITION_VALUE_1  = round(a11/0.088)                 # Value for disabling the torque
                DXL_POS_3_1                   = round(b11/0.088)
                DXL_MINIMUM_POSITION_VALUE_2  = round(229/0.088)           # Dynamixel will rotate between this value
                DXL_MAXIMUM_POSITION_VALUE_2  = round(a22/0.088)            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
                DXL_POS_3_2                   = round(b22/0.088)
                DXL_MINIMUM_POSITION_VALUE_3  = round(342/0.088)           # Dynamixel will rotate between this value
                DXL_MAXIMUM_POSITION_VALUE_3  = round(a33/0.088)            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
                DXL_POS_3_3                   = round(b33/0.088)  
                DXL_MINIMUM_POSITION_VALUE_4  = round(217/0.088)           # Dynamixel will rotate between this value
                DXL_MAXIMUM_POSITION_VALUE_4  = round(a44/0.088)           # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
                DXL_POS_3_4                   = round(b44/0.088)
                DXL_MINIMUM_POSITION_VALUE_5  = 516           # Dynamixel will rotate between this value
                DXL_MAXIMUM_POSITION_VALUE_5  = 944            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
                DXL_MOVING_STATUS_THRESHOLD   = 5                # Dynamixel moving status threshold
                DXL_PROF_VEL_VAL_1            = 10
                DXL_PROF_VEL_VAL_2            = 20
                DXL_PROF_VEL_VAL_3            = 30
                DXL_PROF_VEL_VAL_4            = 30
                DXL_PROF_ACC_VAL_1            = 5
                DXL_PROF_ACC_VAL_2            = 5
                DXL_PROF_ACC_VAL_3            = 5
                DXL_PROF_ACC_VAL_4            = 5
                DXL_D_GAIN_VAL                = 500


                index = 0
                dxl_goal_position_1 = [DXL_MINIMUM_POSITION_VALUE_1, DXL_MAXIMUM_POSITION_VALUE_1, DXL_POS_3_1]
                dxl_goal_position_2 = [DXL_MINIMUM_POSITION_VALUE_2, DXL_MAXIMUM_POSITION_VALUE_2, DXL_POS_3_2]
                dxl_goal_position_3 = [DXL_MINIMUM_POSITION_VALUE_3, DXL_MAXIMUM_POSITION_VALUE_3, DXL_POS_3_3]        # Goal position
                dxl_goal_position_4 = [DXL_MINIMUM_POSITION_VALUE_4, DXL_MAXIMUM_POSITION_VALUE_4, DXL_POS_3_4]        # Goal position
                dxl_goal_position_5 = [DXL_MINIMUM_POSITION_VALUE_5, DXL_MAXIMUM_POSITION_VALUE_5, DXL_MINIMUM_POSITION_VALUE_5]        # Goal position


                # Initialize PortHandler instance
                # Set the port path
                # Get methods and members of PortHandlerLinux or PortHandlerWindows
                portHandler = PortHandler(DEVICENAME)

                # Initialize PacketHandler instance
                # Set the protocol version
                # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
                packetHandler1 = PacketHandler(PROTOCOL_VERSION1)
                packetHandler2 = PacketHandler(PROTOCOL_VERSION2)

                # Open port
                if portHandler.openPort():
                    print("Succeeded to open the port")
                else:
                    print("Failed to open the port")
                    print("Press any key to terminate...")
                    getch()
                    quit()


                # Set port baudrate
                if portHandler.setBaudRate(BAUDRATE):
                    print("Succeeded to change the baudrate")
                else:
                    print("Failed to change the baudrate")
                    print("Press any key to terminate...")
                    getch()
                    quit()
                # Enable Joint 1 Torque
                dxl_comm_result, dxl_error = packetHandler2.write2ByteTxRx(portHandler, DXL1_ID, ADDR_XH_TORQUE_ENABLE, TORQUE_ENABLE)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Joint 1 has been successfully connected")

                # Enable Joint 2 Torque
                dxl_comm_result, dxl_error = packetHandler2.write2ByteTxRx(portHandler, DXL2_ID, ADDR_XM_TORQUE_ENABLE, TORQUE_ENABLE)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Joint 2 has been successfully connected")

                # Enable Joint 3 Torque
                dxl_comm_result, dxl_error = packetHandler2.write1ByteTxRx(portHandler, DXL3_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Joint 3 has been successfully connected")
                # Enable Joint 4 Torque
                dxl_comm_result, dxl_error = packetHandler2.write1ByteTxRx(portHandler, DXL4_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Joint 4 has been successfully connected")
                # Enable Gripper Torque
                dxl_comm_result, dxl_error = packetHandler1.write1ByteTxRx(portHandler, DXL5_ID, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler1.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler1.getRxPacketError(dxl_error))
                else:
                    print("Gripper has been successfully connected")

                #Write Profile Velocity Value For Joint 1
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL1_ID, ADDR_XH_PROFILE_VELOCITY, DXL_PROF_VEL_VAL_1)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Profile Velocity For Joint 1 Value Written")

                #Write Profile Velocity Value For Joint 2
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL2_ID, ADDR_XM_PROFILE_VELOCITY, DXL_PROF_VEL_VAL_2)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Profile Velocity For Joint 2 Value Written")

                #Write Profile Velocity Value For Joint 3
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL3_ID, ADDR_MX_PROFILE_VELOCITY, DXL_PROF_VEL_VAL_3)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Profile Velocity For Joint 3 Value Written")

                #Write Profile Velocity Value For Joint 4
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL4_ID, ADDR_MX_PROFILE_VELOCITY, DXL_PROF_VEL_VAL_4)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Profile Velocity For Joint 4 Value Written")

                #Write Profile Acceleration Value For Joint 1
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL1_ID, ADDR_XH_PROFILE_ACCELERATION, DXL_PROF_ACC_VAL_1)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Profile Acceleration For Joint 1 Value Written")

                #Write Profile Acceleration Value For Joint 2
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL2_ID, ADDR_XM_PROFILE_ACCELERATION, DXL_PROF_ACC_VAL_2)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Profile Acceleration For Joint 2 Value Written")

                #Write Profile Acceleration Value For Joint 3
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL3_ID, ADDR_MX_PROFILE_ACCELERATION, DXL_PROF_ACC_VAL_3)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Profile Acceleration For Joint 3 Value Written")

                #Write Profile Velocity Value For Joint 4
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL4_ID, ADDR_MX_PROFILE_ACCELERATION, DXL_PROF_ACC_VAL_4)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("Profile Acceleration For Joint 4 Value Written")

                #Write D Gain Value For Joint 1
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL1_ID, ADDR_D_GAIN, DXL_D_GAIN_VAL)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("D Gain For Joint 1 Value Written")

                #Write D Gain Value For Joint 2
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL2_ID, ADDR_D_GAIN, DXL_D_GAIN_VAL)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("D Gain For Joint 2 Value Written")

                #Write D Gain Value For Joint 3
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL3_ID, ADDR_D_GAIN, DXL_D_GAIN_VAL)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("D Gain For Joint 3 Value Written")

                #Write D Gain Value For Joint 4
                dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL4_ID, ADDR_D_GAIN, DXL_D_GAIN_VAL)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))
                else:
                    print("D Gain For Joint 4 Value Written")

                while 1:
                    print("Press any key to Move Arm! (or press ESC to End Sequence!)")
                    if getch() == chr(0x1b):
                        break
                    # Write goal position joint 1
                    dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL1_ID, ADDR_XH_GOAL_POSITION, dxl_goal_position_1[index])
                    if dxl_comm_result != COMM_SUCCESS:
                        print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                    elif dxl_error != 0:
                        print("%s" % packetHandler2.getRxPacketError(dxl_error))

                    # Write goal position joint 2
                    dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL2_ID, ADDR_XM_GOAL_POSITION, dxl_goal_position_2[index])
                    if dxl_comm_result != COMM_SUCCESS:
                        print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                    elif dxl_error != 0:
                        print("%s" % packetHandler2.getRxPacketError(dxl_error))

                    # Write goal position joint 3
                    dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL3_ID, ADDR_MX_GOAL_POSITION, dxl_goal_position_3[index])
                    if dxl_comm_result != COMM_SUCCESS:
                        print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                    elif dxl_error != 0:
                        print("%s" % packetHandler2.getRxPacketError(dxl_error))

                    # Write goal position joint 4
                    dxl_comm_result, dxl_error = packetHandler2.write4ByteTxRx(portHandler, DXL4_ID, ADDR_MX_GOAL_POSITION, dxl_goal_position_4[index])
                    if dxl_comm_result != COMM_SUCCESS:
                        print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                    elif dxl_error != 0:
                        print("%s" % packetHandler2.getRxPacketError(dxl_error))


                    while 1:
                        # Read joint 1 present position
                        dxl_present_position_1, dxl_comm_result, dxl_error = packetHandler2.read4ByteTxRx(portHandler, DXL1_ID, ADDR_XH_PRESENT_POSITION)
                        if dxl_comm_result != COMM_SUCCESS:
                            print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                        elif dxl_error != 0:
                            print("%s" % packetHandler2.getRxPacketError(dxl_error))

                        print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL1_ID, dxl_goal_position_1[index], dxl_present_position_1))

                        # Read joint 2 present position
                        dxl_present_position_2, dxl_comm_result, dxl_error = packetHandler2.read4ByteTxRx(portHandler, DXL2_ID, ADDR_XM_PRESENT_POSITION)
                        if dxl_comm_result != COMM_SUCCESS:
                            print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                        elif dxl_error != 0:
                            print("%s" % packetHandler2.getRxPacketError(dxl_error))

                        print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL2_ID, dxl_goal_position_2[index], dxl_present_position_2))

                        # Read joint 3 present position
                        dxl_present_position_3, dxl_comm_result, dxl_error = packetHandler2.read4ByteTxRx(portHandler, DXL3_ID, ADDR_MX_PRESENT_POSITION)
                        if dxl_comm_result != COMM_SUCCESS:
                            print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                        elif dxl_error != 0:
                            print("%s" % packetHandler2.getRxPacketError(dxl_error))

                        print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL3_ID, dxl_goal_position_3[index], dxl_present_position_3))

                        # Read joint 4 present position
                        dxl_present_position_4, dxl_comm_result, dxl_error = packetHandler2.read4ByteTxRx(portHandler, DXL4_ID, ADDR_MX_PRESENT_POSITION)
                        if dxl_comm_result != COMM_SUCCESS:
                            print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                        elif dxl_error != 0:
                            print("%s" % packetHandler2.getRxPacketError(dxl_error))

                        print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL4_ID, dxl_goal_position_4[index], dxl_present_position_4))
                        
                        if not (abs(dxl_goal_position_1[index] - dxl_present_position_1) > DXL_MOVING_STATUS_THRESHOLD):
                            break
                        if not (abs(dxl_goal_position_2[index] - dxl_present_position_2) > DXL_MOVING_STATUS_THRESHOLD):
                            break
                        if not (abs(dxl_goal_position_3[index] - dxl_present_position_3) > DXL_MOVING_STATUS_THRESHOLD):
                            break
                        if not (abs(dxl_goal_position_4[index] - dxl_present_position_4) > DXL_MOVING_STATUS_THRESHOLD):
                            break

                        while 1:
                            print("Press any key Move Gripper! (or press ESC to Quit Gripper Sequence!)")
                            if getch() == chr(0x1b):
                                break
                            # Write Gripper goal position
                            dxl_comm_result, dxl_error = packetHandler1.write4ByteTxRx(portHandler, DXL5_ID, ADDR_AX_GOAL_POSITION, dxl_goal_position_5[index])
                            if dxl_comm_result != COMM_SUCCESS:
                                print("%s" % packetHandler1.getTxRxResult(dxl_comm_result))
                            elif dxl_error != 0:
                                print("%s" % packetHandler1.getRxPacketError(dxl_error))

                            while 1:
                                # Read present position
                                dxl_present_position_5, dxl_comm_result, dxl_error = packetHandler1.read4ByteTxRx(portHandler, DXL5_ID, ADDR_AX_PRESENT_POSITION)
                                if dxl_comm_result != COMM_SUCCESS:
                                    print("%s" % packetHandler1.getTxRxResult(dxl_comm_result))
                                elif dxl_error != 0:
                                    print("%s" % packetHandler1.getRxPacketError(dxl_error))

                                print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (DXL5_ID, dxl_goal_position_5[index], dxl_present_position_5))

                                if not abs(dxl_goal_position_5[index] - dxl_present_position_5) > DXL_MOVING_STATUS_THRESHOLD:
                                    break

                    # Change goal position
                    if index == 2:
                        index = 0
                    else :
                        index += 1

                # Disable Joint 2 Torque
                dxl_comm_result, dxl_error = packetHandler2.write1ByteTxRx(portHandler, DXL2_ID, ADDR_XM_TORQUE_ENABLE, TORQUE_DISABLE)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))

                # Disable Joint 3 Torque
                dxl_comm_result, dxl_error = packetHandler2.write1ByteTxRx(portHandler, DXL3_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))

                # Disable Joint 4 Torque
                dxl_comm_result, dxl_error = packetHandler2.write1ByteTxRx(portHandler, DXL4_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler2.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler2.getRxPacketError(dxl_error))

                # Disable Gripper Torque
                dxl_comm_result, dxl_error = packetHandler1.write1ByteTxRx(portHandler, DXL5_ID, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % packetHandler1.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % packetHandler1.getRxPacketError(dxl_error))

                # Close port
                portHandler.closePort()

            # y_real = ((math.sin(((depth_y/424)*30)))/depth_z)+19
            # z_real = ((math.cos(((depth_y/424)*30)))/depth_z)+19
            # x_real = (math.sqrt((depth_z)**2)-((y_real-19)**2))
            # print(x_real, y_real, z_real)
        elapsed_time = time.time() - timeframe
        fps = frame_id / elapsed_time
        cv2.putText(frame_color, str(round(fps,2)), (10, 50), font, 2, (255, 255, 255), 2) #FPS Value
        cv2.putText(frame_color, "FPS", (220, 50), font, 2, (255, 255, 255), 2) #FPS Label
        cv2.putText(depth_colormap, str(round(fps,2)), (10, 50), font, 2, (255, 255, 255), 2) #FPS Value
        cv2.putText(depth_colormap, "FPS", (220, 50), font, 2, (255, 255, 255), 2) #FPS Label
        cv2.imshow('color', frame_color)
        cv2.imshow('depth', depth_colormap)
        
    # key = cv2.waitKey(1)
    # if key == 27:
    #     break
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
kinect.close()
cv2.destroyAllWindows