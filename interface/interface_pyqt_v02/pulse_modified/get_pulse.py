from pulse_modified.lib.device import Camera
from pulse_modified.lib.processors_noopenmdao import findFaceGetPulse
from pulse_modified.lib.interface import imshow, waitKey, destroyWindow
from cv2 import moveWindow
import argparse
import numpy as np
import datetime
import time
import socket
import sys

class getPulseApp(object):

    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self, cam):
        # Imaging device - must be a connected camera (not an ip camera or mjpeg
        # stream)

        self.cameras = []
        self.selected_cam = 0
        for i in range(3):
            camera = Camera(cam)  # first camera by default
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break
        self.w, self.h = 0, 0
        # Containerized analysis of recieved image frames (an openMDAO assembly)
        # is defined next.

        # This assembly is designed to handle all image & signal analysis,
        # such as face detection, forehead isolation, time series collection,
        # heart-beat detection, etc.

        # Basically, everything that isn't communication
        # to the camera device or part of the GUI
        self.processor = findFaceGetPulse(bpm_limits=[50, 160],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.)

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        #(A GUI window must have focus for these to work)

        self.toggle_state = False

    def toggle_cam(self):
        if len(self.cameras) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)


    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been sucessfully isolated.
        """
        state = self.processor.find_faces_toggle()
        print("face detection lock =", not state)


    def main_loop(self):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        frame = self.cameras[self.selected_cam].get_frame()
        self.h, self.w, _c = frame.shape

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis
        self.processor.run(self.selected_cam)
        # collect the output frame for display
        output_frame = self.processor.frame_out

        # create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        # handle any key presses
        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            for cam in self.cameras:
                cam.cam.release()
            sys.exit()

        if self.toggle_state is False:
            self.toggle_search()
            self.toggle_state = True
            
        return  output_frame

