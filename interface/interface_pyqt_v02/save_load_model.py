# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 19/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-07 10:32:27 +0200 (Wed, 07 Aug 2019)               $: Date of the last commit
# //$Revision:: 100                                                        $: Revision of the last commit

"""
Code for choosing a machine learning model from the models folder. Currently disabled.
"""

import pickle
import time

def save_model(method_name,model):
	"""
	Function for saving a model after fitting.
	:param method_name:
	:param model:
	:return:
	"""
	# Get current time and date
	timestr = time.strftime("%Y%m%d-%H%M%S")
	
	pkl_filename = method_name+ "Model_" +timestr
	
	with open(pkl_filename, 'wb') as file:
	    pickle.dump(model, file)
	    print(method_name, "model saved in the current directory.")

def load_model(loaded_model):
	"""
	Function for loading a model.
	:param loaded_model:
	:return:
	"""
	if loaded_model:

		with open(loaded_model, 'rb') as handle:
			model = pickle.load(handle)

		print('Model loaded.')
		return model

	else:
		print('Model could not be loaded.')
		return
		

