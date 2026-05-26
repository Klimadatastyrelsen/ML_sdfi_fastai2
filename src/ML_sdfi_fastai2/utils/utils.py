import configparser
import json
from pathlib import Path
from torchvision.models.resnet import  resnet34, resnet50, resnet152, resnet18
from torchvision.models.inception import inception_v3
import ML_sdfi_fastai2.pytorch_models.models as pytorch_models
import sys


def get_model(model_name):
    """
    :param model_name: the model name as a string
    :return: a function that creates a fastai model or the model name string for special cases
    """

    if  model_name == "resnet18":
        return resnet18
    elif model_name == "resnet34":
        return resnet34
    elif model_name == "resnet50":
        return resnet50
    elif model_name == "resnet152":
        return resnet152
    elif model_name == "inception_v3":
        return inception_v3
    elif model_name == "simple_convnet":
        return pytorch_models.create_custom_model
    elif model_name in ["efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l", 
                        "efficientnetv2_rw_s.ra2_in1k", "efficientnetv2_rw_m.agc_in1k", 
                        "tf_efficientnetv2_l.in21k", "tf_efficientnetv2_xl.in21k", "resnet50.a1_in1k"]:
        # Using a timm backbone. This will be handled by the wwf.timm_learner
        return model_name
    elif model_name in ["segformer-b0", "segformer-b1", "segformer-b2", "segformer-b3", 
                        "segformer-b4", "segformer-b5"]:
        # SegFormer models will be treated differently
        return model_name
    elif model_name in ["swin-small-upernet","swin-small-upernet", "swin-base-upernet", "swin-large-upernet"]:
        # Swin + UPerNet models
        return model_name
    elif model_name in [
        "convnextv2_tiny_upernet",
        "convnextv2_small_upernet",
        "convnextv2_base_upernet",
        "convnextv2_large_upernet"]:
        return model_name

    else:
        sys.exit("utils.utils.py get_model(model_name) did not recognize model_name: " + str(model_name))


def load_settings_from_config_file(config_file_path):
    """
    :param: config_file_path: e.g "path/to/myexperiment.ini"
    :return: dictionary with all settings needed for training/doing inference, e.g modeltype ,weights to load or dataset to train on
    """

    print("##############################################################################################################################")
    print("######################################### PARSING THE SETTINGS FILE ##########################################################")

    settings_dictionary={}
    parser = configparser.ConfigParser()
    if not Path(config_file_path).is_file():
        sys.exit(config_file_path+ ": is not a file!, did you give the correct path?")
    parser.read(config_file_path)
    sections = parser.sections()
    for section in sections:
        if section =="SUBSETS":
            settings_dictionary["paths_to_subset_files"]=[]
            for key in parser[section]:
                settings_dictionary["paths_to_subset_files"].append(Path(parser[section][key]))
        else:
            print("loading settings in the : "+str(section)+" section")
            for key in parser[section]:
                print("loading value for : "+str(key))
                value_for_key = parser[section][key]

                if key == "model":
                    settings_dictionary[key] = get_model(value_for_key)
                elif key in ["model_to_load","model_used_for_inference"]:
                    if value_for_key == "false":
                        settings_dictionary[key]=json.loads(value_for_key)
                    else:
                        settings_dictionary[key]=Path(value_for_key)
                elif section in ["FOLDERS","DATASET"]:
                    settings_dictionary[key] = Path(value_for_key)
                else:
                    print("key:"+str(key))
                    print("value_for_key:"+str(value_for_key))
                    settings_dictionary[key] = json.loads(str(value_for_key))

    print("######################################### FINNISHED PARSING THE SETTINGS FILE#################################################")
    print("##############################################################################################################################")
    return settings_dictionary

def save_dictionary_to_disk(experiment_settings_dict):
    """
    Saves the content of the dictionary as a json file
    :param experiment_settings_dict: a dictionary
    :return: None
    """
    json_serializable_dictionary={}
    for key in experiment_settings_dict:
        json_serializable_dictionary[key]= str(experiment_settings_dict[key])

    #Store all job configurations as json file in the log folder
    with open(experiment_settings_dict["log_folder"]/Path(experiment_settings_dict["job_name"]+ "_job_dictionary.json"), "w") as out_file:
        json.dump(json_serializable_dictionary, out_file, indent = 6)
