# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 19/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-07 10:32:27 +0200 (Wed, 07 Aug 2019)               $: Date of the last commit
# //$Revision:: 100                                                        $: Revision of the last commit

"""
Code for temperature predictions using a given artificial intelligence model.

Metabolic activiy, weight, height and clothing uses default values at the current state.
"""
 
import warnings
warnings.filterwarnings("ignore")
import get_temp_humid
import userprofile as up
from userprofile import UserProfile
import get_monthly_temp as gmt
import save_load_model as slm
import os
import pandas as pd

import joblib

class EstimateTemp():
    """
    Temperature estimation class, containing default parameters for estimation in init.
    """
    def __init__(self, active_user, model, current_humidity, current_temp, cabin_air_velocity, met, clo):
        self.active_user = active_user
        self.model = model
        self.current_humidity = current_humidity
        self.current_temp = current_temp
        self.sex = 0
        self.met = met
        self.clo = clo
        self.read_monthly_temp = gmt.get_monthly_temp()
        self.cabin_air_velocity = cabin_air_velocity

    def get_sex(self):
        """
        Method for converting the gender string input to 0 or 1 for use in prediction model.
        :return:
        """

        if self.active_user.sex == 'Male':
            self.sex = 1
        else:
            self.sex = 0
        return self.sex

    def estimate_temp(self):
        """
        Method for predicting the optimal temperature. Scaler is provided in the same folder. Height and weight input
        needs to be implemented and converted to BMI.
        :return:
        """
        estim_df = pd.DataFrame({
            "age": self.active_user.age,
            "clo": self.clo,
            "met": self.met,
            "relativehumidity": self.current_humidity,
            "airvelocity": self.cabin_air_velocity,
            "BMI": self.calculate_bmi(),
            "outdoormonthlyairtemperature": self.read_monthly_temp,
            "thermalsensation": 0,
            "sex_Male": self.get_sex(),
        }, index=[0])

        scaler = joblib.load("StandardScaler.save")

        estim_df = scaler.transform(estim_df)

        temperature_prediction = self.model.predict(estim_df)
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Ambiant air temperature during predictions is :", self.current_temp, "°C")
        print("Estimated optimal ambiant air temperature is:", temperature_prediction, "°C")
        return temperature_prediction

    def calculate_bmi(self):
        bmi = self.active_user.weight / self.active_user.height ** 2
        return bmi


def estimate(user, load_model, humidity, temp, air_velocity, met, clo):
    """
    Main estimation function, creates and estimation object and returns the estimated value.
    :param user: A user object.
    :param load_model:
    :param humidity:
    :param temp:
    :param air_velocity:
    :return:
    """
    Temp_pred = EstimateTemp(user, load_model, humidity, temp, air_velocity, met, clo)
    estimation = Temp_pred.estimate_temp()
    return estimation

