# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 19/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-07 10:32:27 +0200 (Wed, 07 Aug 2019)               $: Date of the last commit
# //$Revision:: 100                                                    $: Revision of the last commit

"""
Main code for the thermal comfort interface.
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui
from V6 import Ui_MainWindow

import face_recognition
import cv2
import numpy as np
import uuid
import configparser
import dlib
import math
import time
import threading
import os

import estim_temp
import save_load_model as slm
import get_monthly_temp as gmt
import get_temp_humid
import userprofile
from userprofile import UserProfile
import pulse_modified.get_pulse

def logthread(caller):
    """
    Simple fucntion for logging the threads that functions are running in.
    :param caller: Name of the logged function.
    :return: Prints current thread name and current thread id for a given function.
    """
    print('%-25s: %s, %s,' % (caller, threading.current_thread().name,
                              threading.current_thread().ident))

# Import config file.
config = configparser.ConfigParser()
config.read('config.ini')

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    PyQt GUI class, containing init variables and calls the interface to display when an object is created.
    """
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.show()

        print("Main in:", QThread.currentThread())

        # Create worker objects for different functions to run in parallel.
        self.read_temp_humid = ReadTemperatureHumidity()
        self.read_cam = GetCameraFrame()
        self.pred_temp = PredictTemperature()

        # Create threads for each worker.
        self.thread = QThread(self)
        self.thread1 = QThread(self)
        self.thread2 = QThread(self)

        # Move workers to threads.
        self.read_temp_humid.moveToThread(self.thread)
        self.read_cam.moveToThread(self.thread1)
        self.pred_temp.moveToThread(self.thread2)

        # Start threads.
        self.thread.start()
        self.thread1.start()
        self.thread2.start()

        # Signal/slot connections:
        """ 
        To be able to run different functions in parallel, we need to use threads and PyQt has its own
        threading system. As we are running different functions on different things to make different threads to 
        communicate with the main thread (GUI), we use signal/slot system of the PyQt which allows to send signals 
        between functions. The GUI thread must not run any functions to prevent freezing. A background function is 
        created as a QObject, and then moved to its corresponding thread.
        """
        # Connect text outputs from different workers to text display function(text_output_callback) in GUI.
        self.read_cam.text_output.connect(self.text_output_callback)
        self.read_temp_humid.text_output.connect(self.text_output_callback)
        self.pred_temp.text_output.connect(self.text_output_callback)

        # Connect camera frame signals emitted from detection worker to
        self.read_cam.camera_signal.connect(self.frame_callback)

        # Connect text outputs from functions to text display function(text_output_callback) in GUI.
        self.read_cam.seat2_signal.connect(self.seat2_callback)
        self.read_cam.seat1_signal.connect(self.seat1_callback)

        # Connect temperature and humidity readings to GUI display function.
        self.read_temp_humid.temp_humid_signal.connect(self.temp_humid_callback)

        # Connect temperature and humidity readings to prediction function.
        self.read_temp_humid.temp_humid_signal.connect(self.pred_temp.predict)

        # Connect loaded user profiles emitted after predictions to GUI.
        self.pred_temp.seat1_pred_signal.connect(self.seat1_pred_callback)
        self.pred_temp.seat2_pred_signal.connect(self.seat2_pred_callback)

        # Connect pulse signal.
        self.read_cam.pulse_signal.connect(self.bpm_callback)

        # Connect buttons to various functions.
        self.onoffButton.toggled.connect(self.read_temp_humid.read_temperature_humidity)
        self.onoffButton.toggled.connect(self.read_cam.read_frame)

        self.seat1_button.pressed.connect(self.set_selected_user1)
        self.seat2_button.pressed.connect(self.set_selected_user2)

        self.applyuser_button.pressed.connect(self.apply_user_profile)
        self.undouser_button.pressed.connect(self.undo_user_profile)
        self.saveuser_button.pressed.connect(self.save_user_profile)

        self.verticalSlider.valueChanged.connect(self.air_velocity_value)

        self.cabin_tempincrease_button.pressed.connect(self.increase_manual_temp)
        self.cabin_tempdecrease_button.pressed.connect(self.decrease_manual_temp)
        self.target_temp_display.setText("22 °C")

        self.returnButton.pressed.connect(self.kill_application)

        self.aimodel_comboBox.setCurrentText(config['DEFAULT']['model'])

        # Seat predictions.
        self.seat1_pred = None
        self.seat2_pred = None

        # Seat users.
        self.seat1_user = None
        self.seat2_user = None

        # Active user declaration, called when a seat button is clicked.
        self.active_user = None

        # Manual air temperature values for each seat
        self.seat1_manual_temp = 22
        self.seat2_manual_temp = 22

        # Define temporary user profile to be able to modify/save active user parameters.
        self.temp_user_settings = userprofile.UserProfile()

        # Set default value for metabolic activity.
        self.met = 1

        logthread('mainwin.__init__') # Log the thread activity of the current function.

    def increase_manual_temp(self):
        """
        Function to increase displayed temperature on button click while automatic mode is inactive.
        """
        self.seat1_manual_temp = self.seat1_manual_temp + 1
        self.target_temp_display.setText(str(self.seat1_manual_temp)+ " °C")

    def decrease_manual_temp(self):
        """
        Function to decrease displayed temperature on button click while automatic mode is inactive.
        """
        self.seat1_manual_temp = self.seat1_manual_temp - 1
        self.target_temp_display.setText(str(self.seat1_manual_temp)+ " °C")

    def air_velocity_value(self):
        """
        Function to display and return the value of air velocity slider. Slider max value is 3 and min is 0,
        defined in the UI script.
        """
        value = self.verticalSlider.value()
        self.air_velocity_label.setText(str(value))
        return value

    def set_selected_user1(self):
        """
        Function to set seat 1 user active user when seat button is pressed. Also sets the user settings page and
        displays the predicted temperatures.
        """
        self.active_user = self.seat1_user
        self.set_user_settings_page()
        if self.seat1_pred is not None and self.cabin_automatic_button.isChecked():
            self.target_temp_display.setText(str(self.seat1_pred) + " °C")
        elif self.cabin_automatic_button.isChecked():
            self.target_temp_display.setText("Predicting")

    def set_selected_user2(self):
        """
        Function to set seat 2 user active user when seat button is pressed. Also sets the user settings page and
        displays the predicted temperatures.
        """
        self.active_user = self.seat2_user
        self.set_user_settings_page()
        if self.seat2_pred is not None and self.cabin_automatic_button.isChecked():
            self.target_temp_display.setText(str(self.seat2_pred) + " °C")
        elif self.cabin_automatic_button.isChecked():
            self.target_temp_display.setText("Predicting")

    def set_user_settings_page(self):
        """
        Function to display active user parameters in user settings page, is called when a seat button is pressed.
        """
        self.cabin_username_label.setText(self.active_user.name)
        self.userid_lineEdit.setText(str(self.active_user.uuid))
        self.name_lineEdit.setText(str(self.active_user.name))
        self.age_lineEdit.setText(str(self.active_user.age))
        self.height_lineEdit.setText(str(self.active_user.height))
        self.weight_lineEdit.setText(str(self.active_user.weight))
        self.userphoto_label.setPixmap( QPixmap("{}.jpg".format(str(self.active_user.uuid))).scaled(128, 128, QtCore.Qt.KeepAspectRatio))

    def text_output_callback(self, output):
        """
        Function to display on GUI the text signals emitted from different functions.
        :param output: String containing the text to be displayed.
        """
        self.textoutput.append(output + " [" + time.strftime("%H:%M:%S", time.localtime()) + "]")

    def temp_humid_callback(self, temp, humid):
        """
        Function to display on GUI the temperature and humidity reading signals emitted from reading function.
        :param temp: Current cabin temperature.
        :param humid: Current cabin humidity.
        """
        logthread('mainwin.temp_humid_callback')
        self.currenttempdisplay.setText(str(round(temp, 2)) + "°C")
        self.currenthumiddisplay.setText(str(round(humid, 2)) + "%")

    def frame_callback(self, pixmap):
        """
        Function to display on GUI the frame signals sent from the camera.
        :param pixmap: A camera frame.
        """
        logthread('mainwin.frame_callback')
        width = self.camDisplay.width()
        height = self.camDisplay.height()
        smaller_pixmap = pixmap.scaled(height, width, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.camDisplay.setPixmap(smaller_pixmap)

    def seat1_callback(self, seat1):
        """
            Function to set seat1 button to the user detected on the left side of the camera frame. The button is enabled when
        a signal is sent from user detection function.
        :param seat1: Detected user object.
        """
        self.seat1_user = seat1
        self.seat1_button.setText(seat1.name)
        self.seat1_button.setEnabled(True)

    def seat2_callback(self, seat2):
        """
        Function to set seat2 button to the user detected on the right side of the camera frame. The button is enabled when
        a signal is sent from user detection function.
        :param seat2: Detected user object.
        """
        self.seat2_user = seat2
        self.seat2_button.setText(seat2.name)
        self.seat2_button.setEnabled(True)

    def seat1_pred_callback(self,pred):
        """
        Function to get predicted temperature for the user in seat1. Each time a signal is recieved, the display will
        update the value depending on the chosen seat.
        :param pred: Predicted temperature for seat 1.
        :return: Update the temperature display.
        """
        seat_pred = round(pred,2)
        self.seat1_pred = seat_pred
        if self.seat1_button.isChecked() and self.cabin_automatic_button.isChecked():
            self.target_temp_display.setText(str(self.seat1_pred) + " °C")

    def seat2_pred_callback(self,pred):
        """
        Function to get predicted temperature for the user in seat2. Each time a signal is recieved, the display will
        update the value depending on the chosen seat.
        :param pred: Predicted temperature for seat 2.
        :return: Update the temperature display.
        """
        seat_pred = round(pred,2)
        self.seat2_pred = seat_pred
        if self.seat2_button.isChecked() and self.cabin_automatic_button.isChecked():
            self.target_temp_display.setText(str(self.seat2_pred) + " °C")

    def bpm_callback(self, bpm):
        """
        Function for converting the heart rythm to metabolic activity index. 
        :param bpm: Heart rate in bpm.
        :return: Sets the metabolic activity index.
        """
        logthread('mainwin.bpm_callback')
        self.bpm_lineEdit.setText(bpm + " bpm")
        if self.active_user is not None and self.active_user.sex == 'Male':
            self.met = -4.577 + 0.0858 * float(bpm)
        elif self.active_user is not None:
            self.met = -3.633 + 0.0605 * float(bpm)

    def apply_user_profile(self):
        """
        Function to apply modified user settings to the temporary profile object for future saving.
        """
        self.temp_user_settings.uuid = self.userid_lineEdit.text()
        self.temp_user_settings.name = self.name_lineEdit.text()
        self.temp_user_settings.age = int(self.age_lineEdit.text())
        self.temp_user_settings.height = int(self.height_lineEdit.text())
        self.temp_user_settings.weight = int(self.weight_lineEdit.text())
        self.temp_user_settings.sex = self.sex_comboBox.currentText()
        self.textoutput.append("User settings applied" + " [" + time.strftime("%H:%M:%S", time.localtime()) + "]")

    def undo_user_profile(self):
        """
        Function to undo modified user settings.
        """
        self.cabin_username_label.setText(self.active_user.name)
        self.userid_lineEdit.setText(str(self.active_user.uuid))
        self.name_lineEdit.setText(str(self.active_user.name))
        self.age_lineEdit.setText(str(self.active_user.age))
        self.height_lineEdit.setText(str(self.active_user.height))
        self.weight_lineEdit.setText(str(self.active_user.weight))
        self.sex_comboBox.setCurrentText(self.active_user.sex)


    def save_user_profile(self):
        """
        Function to save modified user settings and load the user with updated settings
        to display the changes on the interface
        """
        userprofile.save_user(self.temp_user_settings)
        self.active_user = userprofile.load_user(self.active_user.uuid)
        try:
            if self.active_user.uuid == self.seat1_user.uuid:
                self.seat1_button.setText(self.active_user.name)
                self.seat1_user = self.active_user
        except:
            pass

        try:
            if self.active_user.uuid == self.seat2_user.uuid:
                self.seat2_button.setText(self.active_user.name)
                self.seat2_user = self.active_user
        except:
            pass

        #Display when settings are saved.
        self.textoutput.append("User settings saved." + " [" + time.strftime("%H:%M:%S", time.localtime()) + "]")

    def kill_application(self):
        """
        A function to terminate the program. At its current state it only closes the interface window and releases the
        camera and it does not terminate the threads.
        :return:
        """
        self.close()
        self.read_cam.video_capture.release()

class ReadTemperatureHumidity(QObject):
    """
    Worker class for humidity and temperature readings.
    """
    #Signal declarations.
    temp_humid_signal = QtCore.pyqtSignal(float, float)
    text_output = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot()
    def read_temperature_humidity(self):
        logthread('ReadTemperatureHumidity.ReadTemperatureHumidity')
        # Send signal containing text.
        self.text_output.emit("Temperature and humidity readings started.")

        # Read temperature and humidity every 5 seconds.
        while True:
            temp = get_temp_humid.get_temp_reading()
            humid = get_temp_humid.get_humid_reading()
            self.temp_humid_signal.emit(temp, humid)
            time.sleep(5)

class GetCameraFrame(QObject):
    """
    Function detect and identify faces. As the faces are identified and object containing the user information is
    sent to the GUI
    """
    def __init__(self):
        super(GetCameraFrame,self).__init__()
        logthread('GetCameraFrame.__init__')
        self.detector = dlib.get_frontal_face_detector() # Face detector.
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # Face landmark detector.
        self.video_capture = cv2.VideoCapture(0) # Initialize camera for capture.
    
    #Signal declarations.
    camera_signal = QtCore.pyqtSignal(object)
    text_output = QtCore.pyqtSignal(str)
    seat1_signal = QtCore.pyqtSignal(object)
    seat2_signal = QtCore.pyqtSignal(object)
    user_found_signal = QtCore.pyqtSignal(str)
    pulse_signal = QtCore.pyqtSignal(str)

    @QtCore.pyqtSlot()
    def send_frame(self, frame):
        """
        Function for converting and emitting captured frames.
        :param frame: Captured camera frame.
        """
        logthread('GetCameraFrame.send_frame')
        clr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimage = QtGui.QImage(clr_frame, clr_frame.shape[1], clr_frame.shape[0],
                              QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.camera_signal.emit(pixmap)
        ret, frame = self.video_capture.read() # Get a frame early to prepare camera.

    @QtCore.pyqtSlot()
    def read_frame(self):

        """
        Main function for face detection/identification and user profile loading.
        """
        logthread('GetCameraFrame.read_frame')
        # Get a reference to webcam #0 (the default one)


        # Create arrays of known face encodings and their names
        self.text_output.emit("User detection started.")

        known_face_encodings = np.loadtxt('face_encodingtest1.txt')
        known_face_uuids = np.load('nom_utilisateurstest1.npy')

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_uuids = []
        face_landmarks_list = []
        detected_faces = []
        user_number = int(config['DEFAULT']['user_number'])
        process_this_frame = True

        self.seat2_obj = None
        self.seat1_obj = None

        # Face identification will start when the number of detected faces matches the user_number specified in the
        # config folder. By default it is set as 2, meaning that the identification will start as soon as faces are
        # detected inside a frame.

        # Variable to check face alignement state, if True while loop will end.
        faces_aligned = False

        # Loop while faces are not aligned, when faces are aligned frame is further processed.
        while not faces_aligned:
            logthread('GetCameraFrame.read_frame.face_alignement_process')
            face_alignements = []

            ret, frame = self.video_capture.read()
            size = frame.shape
            dets = self.detector(frame, 1)

            self.send_frame(frame)

            # For each detected face, i for index, d for face coordinates.
            for i, d in enumerate(dets):

                # Get the landmarks/parts for the face in box d.
                shape = self.predictor(frame, d)

                # 2D image points. If you change the image, you need to change vector
                image_points = np.array([
                    (shape.part(30).x, shape.part(30).y),  # Nose tip
                    (shape.part(8).x, shape.part(8).y),  # Chin
                    (shape.part(36).x, shape.part(36).y),  # Left eye left corner
                    (shape.part(45).x, shape.part(45).y),  # Right eye right corne
                    (shape.part(48).x, shape.part(48).y),  # Left Mouth corner
                    (shape.part(54).x, shape.part(54).y)  # Right mouth corner
                ], dtype="double")

                # 3D model points.
                model_points = np.array([
                    (0.0, 0.0, 0.0),  # Nose tip
                    (0.0, -330.0, -65.0),  # Chin
                    (-225.0, 170.0, -135.0),  # Left eye left corner
                    (225.0, 170.0, -135.0),  # Right eye right corne
                    (-150.0, -150.0, -125.0),  # Left Mouth corner
                    (150.0, -150.0, -125.0)  # Right mouth corner
                ])

                # Camera internals.
                focal_length = size[1]
                center = (size[1] / 2, size[0] / 2)
                camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype="double"
                )

                dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                              dist_coeffs,
                                                                              flags=cv2.SOLVEPNP_ITERATIVE)

                # Project a 3D point (0, 0, 1000.0) onto the image plane.
                # We use this to draw a line sticking out of the nose
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                                 translation_vector,
                                                                 camera_matrix, dist_coeffs)

                frame_to_display = frame

                # Draw face landmark points.
                for p in image_points:
                    cv2.circle(frame_to_display, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                # Coordinates for orientation line.
                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                # Draw orientation line.
                cv2.line(frame_to_display, p1, p2, (255, 0, 0), 2)

                # Emit a frame signal to display.
                self.send_frame(frame_to_display)

                # Check if a face is aligned by comparing the length of the line to the distance between two eyes.
                if math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) <= (
                        math.sqrt((shape.part(36).x - shape.part(45).x) ** 2 + (
                                shape.part(36).y - shape.part(45).y) ** 2)):
                    face_alignements.append(True)
                else:
                    face_alignements.append(False)

            # Check if all detected faces are aligned.
            if len(face_alignements) >= 1 and any(face_alignements):
                print("Face(s) aligned")
                faces_aligned = True
            else:
                print("Face(s) not aligned")
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = frame

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Find all the face encodings in the current frame of video if enough faces are detected in the frame.
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_uuids = []
        face_positions = []

        for idx, face_encoding in enumerate(face_encodings):
            if True:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance= float(config['DEFAULT']['tolerance']))
                face_uuid = str(uuid.uuid4())

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    face_uuid = known_face_uuids[best_match_index]
                    if face_uuid not in detected_faces:
                        detected_faces.append(face_uuid)
                        self.text_output.emit("User detected.")

                # If face is now known, add the new face with an uuid and its encodings to database.
                else:
                    known_face_encodings = np.vstack((known_face_encodings, face_encoding))
                    np.savetxt('face_encodingtest1.txt', known_face_encodings)
                    known_face_uuids = np.append(known_face_uuids, [face_uuid], axis=0)
                    np.save('nom_utilisateurstest1', known_face_uuids)
                    if face_uuid not in detected_faces:
                        detected_faces.append(face_uuid)
                        self.text_output.emit("User detected.")

                face_uuids.append(face_uuid)
                top, right, bottom, left = face_locations[idx]
                top *= 1
                right *= 1
                bottom *= 1
                left *= 1

                # Save the detected face as jpg to the same folder with the uuid name.
                # (In the future a seperate folder will be created).
                crop = frame[top:bottom, left:right]
                cv2.imwrite("%s.jpg" % face_uuids[idx], crop)

                # Check if the detected face is in the left or right side of the frame.
                position_annot = None
                frame_width = self.video_capture.get(3)
                frame_width = frame_width / 2
                position = (right + left) / 2

                if position > frame_width:
                    position_annot = "left"
                else:
                    position_annot = "right"

                # Append the face positions.
                face_positions.append(position_annot)

            else:
                    pass

            # Display the results
            for (top, right, bottom, left), face_uuid, positions in zip(face_locations, face_uuids, face_positions):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Load user profile for left and right seat users if user is detected.
                if positions == "right" and self.seat2_obj is None:
                    self.seat2_obj = userprofile.load_user(face_uuid)
                    self.seat2_signal.emit(self.seat2_obj)
                    self.user_found_signal.emit("Right seat user found.")
                if positions == "left" and self.seat1_obj is None:
                    self.seat1_obj = userprofile.load_user(face_uuid)
                    self.seat1_signal.emit(self.seat1_obj)
                    self.user_found_signal.emit("Left seat user found.")

        self.text_output.emit(str(len(detected_faces)) + " user(s) detected.")

        PulseApp = pulse_modified.get_pulse.getPulseApp(self.video_capture)
        start_time = time.time()
        self.text_output.emit("Pulse detection started.")
        self.pulse_signal.emit("Estimating")
        while True:
            pulse_frame = PulseApp.main_loop()
            if PulseApp.processor.mean_bpm != 0 and len(PulseApp.processor.bpm_list) == 39:
                print(PulseApp.processor.mean_bpm)

                self.pulse_signal.emit(str(round(PulseApp.processor.mean_bpm, 2)))
            self.send_frame(pulse_frame)

class PredictTemperature(QObject):
    """ Worker class for temperature predictions."""
    #Signal declarations.
    seat1_pred_signal = QtCore.pyqtSignal(float)
    seat2_pred_signal = QtCore.pyqtSignal(float)
    text_output = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(PredictTemperature, self).__init__(*args, **kwargs)
        self.started = False
        self.model_folder_path = os.getcwd() + "\\models" # Get the model folder path.
        self.loaded_model = slm.load_model(os.path.join(self.model_folder_path + "\\" + config['DEFAULT']['model'])) # Load defautl model.

    @QtCore.pyqtSlot(float, float)
    def predict(self, temp, humid):
        """
            Function for predicting optimal air temperatures for each user using an artificial intelligence model.
        A user object, humidity, temperature and air velocity values are needed. In the current state, height,
        weight, clothing and metabolic activity values are default.
        :param temp: Cabin temperature reading.
        :param humid: Cabin humidity reading.
        :return: A predicted optimal temperature.
        """

        # Check if the chosen model is different than the default model, if yes load the chosen model.
        if window.aimodel_comboBox.currentText()!= config['DEFAULT']['model'] :
            self.load_model_name = window.aimodel_comboBox.currentText()
            self.loaded_model = slm.load_model(os.path.join(self.model_folder_path +"\\"+ window.aimodel_comboBox.currentText()))

        # If detection button is checked, predict the optimal temperatures and emit the predictions.
        if window.cabin_automatic_button.isChecked():
            if self.started is False:
                self.text_output.emit("Automatic temperature predictions started.")
                self.started = True
            try:
                seat1_user_pred = estim_temp.estimate(window.seat1_user, self.loaded_model, humid, temp,
                                                      window.air_velocity_value()/10, window.met,
                                                      int(window.clothing_comboBox.currentText()))
                self.seat1_pred_signal.emit(seat1_user_pred)

            except:
                    pass

            try:
                seat2_user_pred = estim_temp.estimate(window.seat2_user, self.loaded_model, humid, temp, 
                                                      window.air_velocity_value()/10, window.met,
                                                      int(window.clothing_comboBox.currentText()))
                self.seat2_pred_signal.emit(seat2_user_pred)
            except:
                pass

        if not window.cabin_automatic_button.isChecked():
            if self.started is True:
                self.text_output.emit("Automatic temperature predictions stopped.")
                self.started = False


app = QApplication([])
window = MainWindow()
app.exec_()
