# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 19/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-07 10:32:27 +0200 (Wed, 07 Aug 2019)               $: Date of the last commit
# //$Revision:: 100                                                    $: Revision of the last commit

"""
Code for generating random temperature and humidity values. Can be used in future for getting sensor readings.
"""

from random import triangular

def get_temp_reading():
	"""
	Generate random temperature between 22 and 26.
	:return:
	"""
	x = triangular(22, 26)
	return x

def get_humid_reading():
	"""
	Generate random humidity between 30 and 70.
	:return:
	"""
	y = triangular(30, 70)
	return y
	
	
