# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 14:28:31 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 175                                                    $: Revision of the last commit

"""
Script for plotting model response to varying inputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_variables(scaler, model, method_name, dataset):

	met = np.linspace(10,90,100)
	preded_list = []
	met_list = []

	for mets in met:
	    X_manual = pd.DataFrame({
	        "age" : 30,
	        "clo" : 1,
	        "met": 1.2,
	        "relativehumidity": mets,
	        "airvelocity" : 0.15,
	        "BMI": 25,
	        "outdoormonthlyairtemperature" : 20,
	        "thermalsensation": 0,
	        "sex_Male": 1,
	    }, index=[0])

	    X_manual = scaler.transform(X_manual)

	    preded = model.predict(X_manual)
	    preded_list.append(preded)
	    met_list.append(mets)


	plt.suptitle('Prédictions avec '+ method_name +".", fontsize=14)
	ax1 = plt.subplot(331)
	ax1.set_ylim(20, 30)
	ax1.title.set_text(method_name)
	plt.plot(met_list, preded_list, label=method_name)
	plt.legend(fontsize='xx-small')
	plt.xlabel("Relative humidity (%)")
	plt.ylabel("Predicted temperature(°C)")
	plt.axvline(x=np.percentile(dataset.RH, 25),dashes=(2,2),color='red')
	plt.axvline(x=np.percentile(dataset.RH, 75),dashes=(2,2), color='red', label = "IQR")
	plt.legend(fontsize = 'xx-small')



	met = np.linspace(10,90,100)
	preded_list = []
	met_list = []

	for mets in met:
	    X_manual = pd.DataFrame({
	        "age" : mets,
	        "clo" : 0.8,
	        "met": 1.2,
	        "relativehumidity": 40,
	        "airvelocity" : 0.15,
	        "BMI": 25,
	        "outdoormonthlyairtemperature" : 20,
	        "thermalsensation": 0,
	        "sex_Male": 1,
	    }, index=[0])

	    X_manual = scaler.transform(X_manual)

	    preded = model.predict(X_manual)
	    preded_list.append(preded)
	    met_list.append(mets)

	ax2 = plt.subplot(332)
	ax2.set_ylim(20, 30)
	plt.plot(met_list, preded_list, label=method_name)
	plt.legend(fontsize='xx-small')
	plt.xlabel("Age")
	plt.ylabel("Predicted temperature(°C)")
	plt.axvline(x=np.percentile(dataset.AGE, 25),dashes=(2,2),color='red')
	plt.axvline(x=np.percentile(dataset.AGE, 75),dashes=(2,2),color='red', label = "IQR")
	plt.legend(fontsize = 'xx-small')

	met = np.linspace(0.1,3,50)
	preded_list = []
	met_list = []

	for mets in met:
	    X_manual = pd.DataFrame({
	        "age" : 30,
	        "clo" : mets,
	        "met": 1.2,
	        "relativehumidity": 40,
	        "airvelocity" : 0.15,
	        "BMI": 25,
	        "outdoormonthlyairtemperature" : 20,
	        "thermalsensation": 0,
	        "sex_Male": 1,
	    }, index=[0])

	    X_manual = scaler.transform(X_manual)

	    preded = model.predict(X_manual)
	    preded_list.append(preded)
	    met_list.append(mets)

	ax3 = plt.subplot(333)
	ax3.set_ylim(20, 30)
	plt.plot(met_list, preded_list, label=method_name)
	plt.xlabel("Clothing")
	plt.ylabel("Predicted temperature(°C)")
	plt.axvline(x=np.percentile(dataset.CLO, 25),dashes=(2,2),color='red')
	plt.axvline(x=np.percentile(dataset.CLO, 75),dashes=(2,2),color='red', label = "IQR")
	plt.legend(fontsize = 'xx-small')

	met = np.linspace(0.1,3,50)
	preded_list = []
	met_list = []

	for mets in met:
	    X_manual = pd.DataFrame({
	        "age" : 30,
	        "clo" : 0.8,
	        "met": mets,
	        "relativehumidity": 40,
	        "airvelocity" : 0.15,
	        "BMI": 25,
	        "outdoormonthlyairtemperature" : 20,
	        "thermalsensation": 0,
	        "sex_Male": 1,
	    }, index=[0])

	    X_manual = scaler.transform(X_manual)

	    preded = model.predict(X_manual)
	    preded_list.append(preded)
	    met_list.append(mets)

	ax4 = plt.subplot(334)
	ax4.set_ylim(20, 30)
	plt.plot(met_list, preded_list, label=method_name)
	plt.xlabel("Metabolic Activity")
	plt.ylabel("Predicted temperature(°C)")
	plt.axvline(x=np.percentile(dataset.MET, 25),dashes=(2,2),color='red')
	plt.axvline(x=np.percentile(dataset.MET, 75),dashes=(2,2),color='red', label = "IQR")
	plt.legend(fontsize='xx-small')

	met = np.linspace(0,1,100)
	preded_list = []
	met_list = []

	for mets in met:
	    X_manual = pd.DataFrame({
	        "age" : 30,
	        "clo" : 0.8,
	        "met": 1.2,
	        "relativehumidity": 40,
	        "airvelocity" : mets,
	        "BMI": 25,
	        "outdoormonthlyairtemperature" : 20,
	        "thermalsensation": 0,
	        "sex_Male": 1,
	    }, index=[0])

	    X_manual = scaler.transform(X_manual)

	    preded = model.predict(X_manual)
	    preded_list.append(preded)
	    met_list.append(mets)

	ax5 = plt.subplot(335)
	ax5.set_ylim(20, 30)
	plt.plot(met_list, preded_list, label=method_name)
	plt.xlabel("Air velocity(m/s)")
	plt.ylabel("Predicted temperature(°C)")
	plt.axvline(x=np.percentile(dataset.AV, 25),dashes=(2,2),color='red')
	plt.axvline(x=np.percentile(dataset.AV, 75),dashes=(2,2),color='red', label = "IQR")
	plt.legend(fontsize='xx-small')


	met = np.linspace(10,40,50)
	preded_list = []
	met_list = []

	for mets in met:
	    X_manual = pd.DataFrame({
	        "age" : 30,
	        "clo" : 0.8,
	        "met": 1.2,
	        "relativehumidity": 40,
	        "airvelocity" : 0.15,
	        "BMI": mets,
	        "outdoormonthlyairtemperature" : 20,
	        "thermalsensation": 0,
	        "sex_Male": 1,
	    }, index=[0])

	    X_manual = scaler.transform(X_manual)

	    preded = model.predict(X_manual)
	    preded_list.append(preded)
	    met_list.append(mets)

	ax6 = plt.subplot(336)
	ax6.set_ylim(20, 30)
	plt.plot(met_list, preded_list, label=method_name)
	plt.xlabel("BMI(kg/m^2)")
	plt.ylabel("Predicted temperature(°C)")
	plt.axvline(x=np.percentile(dataset.BMI, 25),dashes=(2,2),color='red')
	plt.axvline(x=np.percentile(dataset.BMI, 75),dashes=(2,2),color='red', label = "IQR")
	plt.legend(fontsize='xx-small')


	met = np.linspace(-10,50,100)
	preded_list = []
	met_list = []

	for mets in met:
	    X_manual = pd.DataFrame({
	        "age" : 30,
	        "clo" : 0.8,
	        "met": 1.2,
	        "relativehumidity": 40,
	        "airvelocity" : 0.1,
	        "BMI": 25,
	        "outdoormonthlyairtemperature" : mets,
	        "thermalsensation": 0,
	        "sex_Male": 1,
	    }, index=[0])

	    X_manual = scaler.transform(X_manual)

	    preded = model.predict(X_manual)
	    preded_list.append(preded)
	    met_list.append(mets)

	ax7 = plt.subplot(337)
	ax7.set_ylim(20, 30)
	plt.plot(met_list, preded_list, label=method_name)
	plt.xlabel("Outdoor monthly temperature(°C)")
	plt.ylabel("Predicted temperature(°C)")
	plt.axvline(x=np.percentile(dataset.OTMP, 25),dashes=(2,2),color='red')
	plt.axvline(x=np.percentile(dataset.OTMP, 75),dashes=(2,2),color='red', label = "IQR")
	plt.legend(fontsize='xx-small')


	met = [0,1]
	preded_list = []
	met_list = []

	for mets in met:
	    X_manual = pd.DataFrame({
	        "age" : 30,
	        "clo" : 0.8,
	        "met": 1.2,
	        "relativehumidity": 40,
	        "airvelocity" : 0.15,
	        "BMI": 25,
	        "outdoormonthlyairtemperature" : 20,
	        "thermalsensation": 0,
	        "sex_Male": mets,
	    }, index=[0])

	    X_manual = scaler.transform(X_manual)

	    preded = model.predict(X_manual)
	    preded_list.append(preded)
	    met_list.append(mets)

	ax8 = plt.subplot(338)
	ax8.set_ylim(20, 30)
	plt.plot(met_list, preded_list, label=method_name)
	plt.xlabel("Sex")
	plt.ylabel("Predicted temperature(°C)")
	plt.legend(fontsize='xx-small')

	plt.show()
