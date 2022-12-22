import os
from numpy import *
from math import *
import numpy as np

st_x = 0
st_y = 15
st_z = 60

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
x_3, y_3, z_3, r_x_1, r_y_1, r_z_1, r_x_2, r_y_2, r_z_2, r_x_3, r_y_3, r_z_3 = forward_kinematics(a11, a22, a33, a44)
a11 = a11*-1
a22=a22+92
a33=(-a33)+169.1
a44=a44+200

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
DXL_POS_3_1                   = round(0.1/0.088)
DXL_MINIMUM_POSITION_VALUE_2  = round(229/0.088)           # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE_2  = round(a22/0.088)            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
DXL_POS_3_2                   = round(150/0.088)
DXL_MINIMUM_POSITION_VALUE_3  = round(342/0.088)           # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE_3  = round(a33/0.088)            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
DXL_POS_3_3                   = round(226/0.088)  
DXL_MINIMUM_POSITION_VALUE_4  = round(217/0.088)           # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE_4  = round(a44/0.088)           # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
DXL_POS_3_4                   = round(124/0.088)
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
dxl_goal_position_1 = [DXL_MINIMUM_POSITION_VALUE_1, DXL_MAXIMUM_POSITION_VALUE_1]
dxl_goal_position_2 = [DXL_MINIMUM_POSITION_VALUE_2, DXL_MAXIMUM_POSITION_VALUE_2]
dxl_goal_position_3 = [DXL_MINIMUM_POSITION_VALUE_3, DXL_MAXIMUM_POSITION_VALUE_3]        # Goal position
dxl_goal_position_4 = [DXL_MINIMUM_POSITION_VALUE_4, DXL_MAXIMUM_POSITION_VALUE_4]        # Goal position
dxl_goal_position_5 = [DXL_MINIMUM_POSITION_VALUE_5, DXL_MAXIMUM_POSITION_VALUE_5]        # Goal position


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
    if index == 1:
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


