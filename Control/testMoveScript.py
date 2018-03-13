#
#  ******************************************************************************
#  * @file     testMoveScript.py
#  * @author   RBRO/PJ-IU
#  * @version  V1.1.0
#  * @date     06-Mar-2018 GMOIS
#  * @modified -
#  * @brief    This file contains the main function for executing vision-based
#              navigation.
#  ******************************************************************************
#

import threading,time,sys
# Module containing parking computation and moving actions
import movingScript
# Module required for using serial communication 
import SerialHandler

## Main function.
#  @param       None
#  @return      None 
def moveManeuver():
    # TODO KeyboardInterrupt for exiting application

    # TODO Nandor
    # Create object for getting data from camera

    # Start thread for getting camera data
    # xxx.start()

    # Create method objects for getting distance and frame number
    # function_distance = xxx.getDistance
    # function_angle = xxx.getFrameNo

    # Instantiates an object for the serial connection used to communicate with Nucleo. 
    # By default, the communication port  is '/dev/ttyACM0' and the baud rate is 460800.
    serialHandler=SerialHandler.SerialHandler()
    # Start the reader thread, which automatically saves the received messages from the Nucleo. 
    # This thread deactivates the waiting state and it  automatically calls a callback function.
    serialHandler.startReadThread()
    
    # Create an object for the movingState thread 
    moving = movingStateClass(serialHandler,function_distance,function_angle)

    # Start thread for controlling the car based on video data
    moving.start()

    # Stop the camera thread
    #xxx.xxx.stop()

    # Stop the moving thread
    moving.Stop()

    # Delete the waiters
    moving.deleteWaiters()

    # Closes all the ports that were opened and all thread that were started.
    serialHandler.close()

# Code only executed when running the module as a program 
#     and not when imported as a module for calling the functions themselves.
if __name__ == "__main__":
    moveManeuver()    