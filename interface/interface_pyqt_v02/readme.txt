Run main.py.

Face orientation is also detected in the captured frame. Face recognition process will start when the faces are oriented towards the camera which will thus increase the face recognition performance.

Metabolic activity estimation is implemented.

Scripts:

dict_obj.py => converts dictionary to object and vice-versa.
estim_temp.py => predicts air temperature using IA model.
get_monthly_temp.py => reads monthly air temperature from csv file.
get_temp_humid.py => generates random temperature and humidity readings.
main.py => main script to run for the interface.
resources_rc.py => contains images to display in the interface.
save_load_model.py => script for saving and loading IA models.
userprofile.py => script for creating/loading/saving user profiles.
interface_frame.py => script containing the graphical interfaca frame.

Other files:

face_encodingtest1.txt => face encodings database.
haarcascade_frontal_face_alt.xml => needed to detect face landmarks.
KNeighborsRegressorModel => machine learning model.
nom_utilisateurtest1.npy => user uuid database.
shape_predictor_5_face_landmarks.dat => face landmark prediction data.
shape_predictor_68_face_landmarks.dat => face landmark prediction data.
StandardScaler.save => scaler to scale input values for predictions.
TemperaturesMensuelles.csv => file containing monhtly air temperatures.
user_db.json => database containing user profiles assiciated to respective uuids.
pulse_modified => folder containing necessary files for heart rate estimations.