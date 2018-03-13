#
#  ******************************************************************************
#  * @file     movingScript.py
#  * @author   RBRO/PJ-IU
#  * @version  V1.1.0
#  * @date     06-Mar-2018 GMOIS
#  * @modified -
#  * @brief    This file contains the class definition for the moving actions
#  *           methods depending on distance between the center of the car and
#  *           the center of the lane. Simple proportional actions are generated.
#  ******************************************************************************
#

import sys
import math
from threading import Thread
import time


##  movingStateClass class. 
#
#  It is used for computing the maneuvers required for moving the car depending on the 
#      data received from the vision module. We should receive data regarding the distance 
#      between the center of the car and the number of the frame. If no new data is received
#      for a second, then car stops and waits for new data.
class movingStateClass(Thread):
    
    # Distance between the center of the car and the center of the lane
    distance_to_center = 0
    # Current frame no.
    frame_no = -1
    # Previous frame no.
    prev_frame_no = 0
    # Angle for next move
    angle = 0.0
    # PWM value for next move
    speed = 0.0
    # TODO Establish values for the factors below
    # Factors for getting angle and speed
    factor_angle = 1
    factor_speed = 1 

    # perform the first move following the first bezier
    ev1=threading.Event()

    # Flag indicating running state
    RUN = True

    ## Constructor.
	#  @param self           The object pointer.
    #  @param serialHandler  serialHandler object.
    #  @param function       Function for getting distance to center of the lane
    #  @param function_frame Function for getting frame number
    def __init__(self,serialHandler,function,function_frame):
        self.serialHandler = serialHandler
        # Function that gets position from GPS
        self.getterFunction = function
        self.getterFunctionFrame = function_frame

        # Add event keys
        self.serialHandler.readThread.addWaiter("MCTL",self.ev1)
        self.serialHandler.readThread.addWaiter("BRAK",self.ev1)

        # Activate the PID
        self.activatePID()

        Thread.__init__(self) 
  
    ## Function that performs the actual moves. Moving action is continuous and stopped only if
    #           data not received for more than one second.
	#  @param self         The object pointer.
    #  @details            UNTESTED        
    def run(self):

        # Perform actions while turn
        while self.RUN:
            # If current frame differs from the previous and it is not -1
            if (self.frame_no != self.prev_frame_no) and (self.frame_no != -1):
                # If distance greater than threshold steer right or left
                if self.distance_to_center > self.threshold :
                    self.angle = self.factor_angle * self.distance_to_center
                    self.speed = 7 + self.factor_speed * abs(self.distance_to_center)
                # If towards lane center, continue straight
                else:
                    self.angle = 0.0
                    self.speed = 10.0

                # Perform moving action
                self.makeArc(self.serialHandler,self.speed,self.angle,ev1)

            else:
                # Push brake
                self.pushBrake(self.serialHandler,angle,ev1)
            
            # Update frame number value
            self.prev_frame_no = self.frame_no

            # Clear the event
            self.ev1.clear

            # Should let the car move a bit backwards. Then get new data.
            time.sleep(1)

    ## Method for starting server negotiation process.
    #  @param self          The object pointer.
    def start(self):
        self.RUN = True
        super(movingStateClass,self).start()

    ## Method for stopping server negotiation process.
    #  @param self          The object pointer.
    def stop(self):
        self.RUN = False

    ## Function that deletes the waiters.
	#  @param self         The object pointer.
    def deleteWaiters(self)
        self.serialHandler.readThread.deleteWaiter("BRAK",self.ev1)
        self.serialHandler.readThread.deleteWaiter("MCTL",self.ev1)
 
    ## Function that gives the command to the motor.
	#  @param self         The object pointer.
    #  @param serialHandler Serial communication object.
    #  @param pwm           pwm value (speed).
    #  @param angle         Steering angle.
    #  @param ev1           Threading event.
    #
    def makeArc(self,serialHandler,pwm,angle,ev1):
        sent=serialHandler.sendMove(pwm,angle)
        if sent:
            print ( " 2 ")
            ev1.wait()
            print("Confirmed")
            ev1.clear()             
        else:
            print("Sending problem MCTL")       
        time.sleep(0.2) 

    ## Function that stops the car.
	#  @param self         The object pointer.
    #  @param serialHandler Serial communication object.
    #
    def pushBrake(self,serialHandler,angle,ev1):
        angle = float(angle)
        sent=serialHandler.sendBrake(angle)
        if sent:
            print ( " 3 ")
            ev1.wait()
            print("Confirmed")
            ev1.clear()               
        else:
            print("Sending problem BRAKE")       
        time.sleep(0.2) 

    ## Function that activates the PID.
	#  @param self         The object pointer.
    #
    def activatePID(self):
        
        # Instantiate an event object, which will be unblocked, when the message was sent to Nucleo.
        pid_activation_event = threading.Event()
        # Attach the event object to the key words. 
        # It will unblock the event object, after the response was received with the specified key word. 
        self.serialHandler.readThread.addWaiter("PIDA",pid_activation_event)
        # Send activate PID command over serial
        sent = self.serialHandler.sendPidActivation(True)
        # Display debug message depending on operation success.
        if sent:
            isConfirmed=pid_activation_event.wait(timeout=1)
            if(isConfirmed):
                print("Confirmed the PID activation commmand!")
            else:
                raise ConnectionError('Response', 'Response was not received!')
            pid_activation_event.clear()
        # Delete the event object attached to the key work.
        self.serialHandler.readThread.deleteWaiter("PIDA",pid_activation_event)