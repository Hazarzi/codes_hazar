# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 19/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-07 10:32:27 +0200 (Wed, 07 Aug 2019)               $: Date of the last commit
# //$Revision:: 100                                                    $: Revision of the last commit

"""
Code containing the userprofile class and functions for saving and loading the userprofile objects inside a database.
"""

import msvcrt
import json
import uuid

# db leri script basinda cagir, dict call fonksiyonu yarat ve print etsin succesfully called veya loaded gibisinden, pro
# fil aktife gecincede mesajla onay versin actife gecti felan diye(if else ile check et, her etabin check edilmesi gerekiyo
# islem yapildiktan sonra yoksa error vermesi lazim

def convert_to_dict(obj):
    """
    	A function takes in a custom object and returns a dictionary representation of the object.
	This dict representation includes meta data such as the object's module and class names.
    :param obj:
    :return:
    """

    #  Populate the dictionary with object meta data
    obj_dict = {
        "__class__": obj.__class__.__name__,
        "__module__": obj.__module__
    }

    #  Populate the dictionary with object properties
    obj_dict.update(obj.__dict__)

    return obj_dict


def dict_to_obj(our_dict):
    """
        Function that takes in a dict and returns a custom object associated with the dict.
    This function makes use of the "__module__" and "__class__" metadata in the dictionary
    to know which object type to create.
    :param our_dict:
    :return:
    """
    if "__class__" in our_dict:
        # Pop ensures we remove metadata from the dict to leave only the instance arguments
        class_name = our_dict.pop("__class__")

        # Get the module name from the dict and import it
        module_name = our_dict.pop("__module__")

        # We use the built in __import__ function since the module name is not yet known at runtime
        module = __import__(module_name)

        # Get the class from the module
        class_ = getattr(module, class_name)

        # Use dictionary unpacking to initialize the object
        obj = class_(**our_dict)
    else:
        obj = our_dict
    return obj

class UserProfile(object):
    """
    A class for user profile objects.
    """

    def __init__(self,uuid=None, name=None, age=None, sex=None, height=None, weight=None):
        """
        Declaration of class parameters = inputs for predictions.
        :param uuid:
        :param name:
        :param age:
        :param sex:
        :param height:
        :param weight:
        """
        self.uuid = uuid

        if name is None:
            self.name = 'Anonymous'
        else:
            self.name = name

        if sex is None:
            self.sex = 'Male'
        else:
            self.sex = sex

        if age is None:
            self.age = 35
        else:
            self.age = age

        if height is None:
            self.height = 170
        else:
            self.height = height

        if weight is None:
            self.weight = 70
        else:
            self.weight = weight

        #self.BMI
    def set_uuid(self,uuid):
        self.uuid = uuid
    def set_name(self):
        self.name = input("Name pls:")
    def set_age(self):
        self.age = input("Age pls:")

def save_user(object_to_save):
    """
    Function that takes a user profile object than saves it to the database.
    :param object_to_save:
    :return:
    """
    with open('user_db.json' , 'r') as file:
        user_data = json.load(file)
    fetched_user = dict_to_obj(user_data.get(object_to_save.uuid))
    fetched_user.age = object_to_save.age
    fetched_user.name = object_to_save.name
    fetched_user.sex = object_to_save.sex
    fetched_user.weight = object_to_save.weight
    fetched_user.height = object_to_save.height
    user_data[object_to_save.uuid] = convert_to_dict(fetched_user)
    with open('user_db.json', 'w') as file:
        json.dump(user_data, file)
    print('User Saved.')




def load_user(get_detected_uuid):
    """
    Function to load a user profile object for a given uuid.
    :param get_detected_uuid:
    :return:
    """

    while True:
        with open('user_db.json', 'r') as file:
            user_check = json.load(file)
        if get_detected_uuid in user_check:
            print('\nUser found. Retrieveing database...\n')
            with open('user_db.json' , 'r') as file:
                user_data = json.load(file)
            active_user = dict_to_obj(user_data.get(get_detected_uuid))
            print("Welcome,", active_user.name)
            return active_user
        else:
            active_user = UserProfile()
            active_user.set_uuid(get_detected_uuid)
            print("Using Anonymous Profile")

            with open('user_db.json', 'r') as file:
                user_data = json.load(file)
                user_data[get_detected_uuid] = convert_to_dict(active_user)
            with open('user_db.json', 'w') as file:
                json.dump(user_data, file)
                
            print('User Saved.')
            
            return active_user
    return active_user
