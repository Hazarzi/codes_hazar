# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 19/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-07 10:32:27 +0200 (Wed, 07 Aug 2019)               $: Date of the last commit
# //$Revision:: 100                                                    $: Revision of the last commit

"""
Code for getting monthly air temperature from a csv file.
"""

import pandas as pd
import datetime

def get_monthly_temp():
	"""
	Reads the csv table containing monthly temperatures.
	:return: 
	"""
	data = pd.read_csv("TemperaturesMensuelles.csv")
	d = datetime.date.today()
	month = int('%02d' % d.month)
	outdoormonthlytemp = data.iloc[month-1]['Temp']
	return outdoormonthlytemp

