#!/usr/bin/env python
# coding: utf-8

"""
Functionality for creating dataloaders for a semantic segmentation-dataset
Main function to call from other modules is  get_dataset(experiment_settings_dict)


"""

#this might fix problems with: OSError: [Errno 9] Bad file descriptor  (https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import ML_sdfi_fastai2.utils.utils as sdfi_utils
from fastcore.xtras import Path
import rasterio
import albumentations as A
from fastai.callback.hook import summary
from fastai.data.block import DataBlock
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import get_image_files, FuncSplitter, Normalize
from fastai.layers import Mish
from fastai.losses import BaseLoss
from fastai.optimizer import ranger
from fastai.torch_core import tensor
from fastai.vision.augment import aug_transforms
from fastai.vision.core import PILImage, PILMask
from fastai.vision.data import ImageBlock, MaskBlock, imagenet_stats
import torchvision.transforms as tfms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import sys
from torchvision.models.resnet import resnet34
import torch
import argparse
from albumentations import ShiftScaleRotate, RandomShadow
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.vision import *
import time
import pathlib
from fastai.data.external import untar_data, URLs
import ML_sdfi_fastai2.transforms.sdfi_transforms as sdfi_transforms
import ML_sdfi_fastai2.ImageBlockReplacement as ImageBlockReplacement

'''
def get_channel(channel,extra_channel_shape ,path):
    """
    :param channel: the kind of channel to ad e.g "nir" (near infra-read) or "d" (depth/height)
    :param extra_channel_shape: the shape of the channel to add
    :param path: path to image we want an extra channel for
    :return: the channel as numpy array
    """
    path = pathlib.Path(path)
    use_dummy_values= False
    if use_dummy_values:
        new_channel=np.zeros(extra_channel_shape)
    else:
            
    
        channel_file_path = path.parent.parent /pathlib.Path(channel) / path.name

        new_channel = np.array(Image.open(channel_file_path))

        
        
        new_channel_max= new_channel.flatten().max()
        new_channel_min= new_channel.flatten().min()
        min_max_range = new_channel_max-new_channel_min


        #handle_divide_by_255 should most probably be set to false , but this kode here untill the excact way /255 -mean /std is aplied is figured out
        handle_divide_by_255 = True
        if handle_divide_by_255:
            new_channel =(((new_channel-new_channel_min)/min_max_range))*255 # make in range[0-255]
        else:
            new_channel =((new_channel-new_channel_min)/min_max_range)-0.5 # make in range[-05,0.5]
        
        
        
        #fastais normalization pipeline is a bit weird, I really need ot figure out exactly how it works. e.g where is mean, std and /255 applied. and how should I handle this with new type of data
        #print("NEW CHANNEL")
        #input(new_channel)

    return new_channel
'''



'''
def open_img(path,experiment_settings_dict,channels_to_zero_out=[]):
    """
    :param path: path to image 
    :param resize:
    :param experiment_settings_dict:
    :channels_to_zero_out: list of  indexes to channels that will be multiplied with 0 (used to verify that model use the values in the channelæ)
    :return:
    



    images should be normalizes like this
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683/19
    https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
    output[channel] = (input[channel] - mean[channel]) / std[channel]
    """
    
    channels= experiment_settings_dict["channels"]

    #     use rasterio to open image as numpy array and rescale to 0-1
    #     you may need to change this if you have values above 255
    img_as_array = rasterio.open(path).read().astype('float32')# looks like division by 255 is done in batch_tfms allready but I dont see how! /255.0 (most probably by the astype('float32'))
    if "divide_by_255" in experiment_settings_dict and experiment_settings_dict["divide_by_255"]:
        img_as_array = img_as_array/255.
        

    if not "nir" in channels and img_as_array.shape[0] ==4:
        print("warning image includes nir but nir is not mentioned as channel")
        print("removing nir channel")
        img_as_array = img_as_array[1:]
        assert img_as_array.shape[0] ==3
        
        
        
    #nir_channel= img_as_array[-1]
    #print(nir_channel.max())
    #print(nir_channel.min())
    #print(np.mean(nir_channel))
    #print(np.std(nir_channel))





    nir_mean= 0.40779018 #based on a sample size of 1! OBS! UNKNOWN
    nir_std=0.15176418  #based on a sample size of 1! OBS! UNKNOWN
    #mean_normalized=(img_as_array.transpose()-np.array([0.485, 0.456, 0.406,nir_mean])).transpose()
    #print("mean_normalized : "+str(mean_normalized ))

    do_manual_normalisation= True # might be that this is done inside fastai somwhere automaticaly!
    if do_manual_normalisation:

        if img_as_array.shape[0] ==4:
            #normalising rgb-nir image
            mean_and_std_normalized=(img_as_array.transpose()-np.array([0.485, 0.456, 0.406,nir_mean],dtype=np.float32)/np.array([0.229, 0.224, 0.225,nir_std],dtype=np.float32) ).transpose()
        else:
            mean_and_std_normalized=(img_as_array.transpose()-np.array([0.485, 0.456, 0.406],dtype=np.float32)/np.array([0.229, 0.224, 0.225],dtype=np.float32) ).transpose()


        img_as_array=mean_and_std_normalized
    #print("mean_and_std_normalized : "+str(mean_and_std_normalized ))


    

    
    
    


    #no need to ad nir-channel since it gets loaded automatically by rasterio.open
    #We can use the following procedure for other kinds of channels, e.g lidar 
    ad_extra_channel= True
    if ad_extra_channel:
        #remove the normal channels from the list in order to get the extra channels
        extra_channels= list(set(channels) -set(["r","g","b","nir"]))
        for channel in extra_channels:
            #input("adding "+str(channel)+" as extra channel")

            extra_channel_shape= [1]+list(img_as_array.shape[1:])
            new_channel=get_channel(channel,extra_channel_shape = extra_channel_shape,path=path)
            #print(img_as_array.shape)
            #print(new_channel.shape)
            
            img_as_array=np.array(np.vstack((img_as_array,new_channel.reshape([1]+list(new_channel.shape)))),dtype=np.float32)


    
    #verify that we now have the correct number of channels
    assert len(channels) ==img_as_array.shape[0]
    
    #if we should set a channel to zero we do it now
    #we only want to do this to sanity chack that the different channels are used
    if "drop_channels" in experiment_settings_dict:
        channels_to_drop = experiment_settings_dict["drop_channels"]
        for i_chanel_to_drop in channels_to_drop:
            verbose = True
            if verbose:
                print("###########################################################################################################")
                print("image shape : "+str(img_as_array.shape))
                print("droppping channel: "+str(i_chanel_to_drop))
               
                print("...")
            #zeroing the nir channel
            img_as_array[i_chanel_to_drop,:,:] = 0
            

    

    #     convert numpy array to tensor
    
    img_as_tensor = torch.from_numpy(img_as_array)
    if len(channels_to_zero_out)>0:
        for channel_to_zero in channels_to_zero_out:
            img_as_tensor[channel_to_zero]=0


    #In order to tell images from labels we cast all images to a special class (ineherits from torch.tensor )
    return sdfi_transforms.MultichannelImage( img_as_tensor)
'''


# we can't use the built in fastai augmentations as they expect 3 channel images so we are using Albumentations instead
# https://github.com/albumentations-team/albumentations
# add as many as you want but these are executed on CPU so can be slow...
'''
transform_list = [A.RandomBrightnessContrast(p=1,brightness_limit=.2),
                  A.RandomRotate90(p=0.5),
                  A.HorizontalFlip(p=.5),
                  A.VerticalFlip(p=.5),
                  A.Blur(p=.1),
                  A.Rotate(p=0.5,limit = 10)
                  ]

'''
#sdfi_tfm = [sdfi_transforms.AlbumentationsTransform(ShiftScaleRotate(p=0.5))]




'''
# apply the augmentations in a loop
def aug_tfm(tensor,experiment_settings_dict):
    """
    :param tensor: image as a tensor
    :param experiment_settings_dict: dictionary with settings
    :return: transformed version of tensor

    Data augmentation during training

    """
    #     this function is used for both images and labels so check the count of the input tensor
    #     if the count is above 1 its not a label so apply the augmentations
    #print("label and data both goa through transformation")
    #print(type(tensor))


    #since sdfes lables not are tensors but fastai fastai.vision.core.PILMask ,we onoy apply the transforms if type is torch.Tensor
    #OBS! if we want to aply rotations both labels and images need to be transformed!!

    #If the dictionary includes a for transforms, read it and apply correct transforms
    use_augmentation = experiment_settings_dict["transforms"]  # The flips and roations must be done on the labels also! but brightnes and such should only be done on image. the solution below can not work properly!
    input(use_augmentation)

    if use_augmentation:


        sdfi_tmf = sdfi_transforms.SegmentationAlbumentationsTransform()


    """

    if use_augmentation and type(tensor) == torch.Tensor:#tensor.nelement()>1:

        
        Transfomrs from internet that does not work! (we need to rotate both labels and iamges, but only images should change color etc)
        #         convert tensor into numpy array and reshape it for Albumentations
        np_array = np.array(tensor.permute(1,2,0))
        #        apply each augmentation
        for transform in transform_list:
            np_array = transform(image=np_array)['image']
        #         some augmentations may shift the values outside of 0-1 so clip them
        np_array = np.clip(np_array, 0, 1)
        #        rearrange image to tensor format
        array_arange = np_array.transpose(2,0,1)
        #        convert back to tensor
        tensor = torch.from_numpy(array_arange)
        #print("apllied the trasnfomration")

        """


    return tensor
'''
'''
def get_transform_function(experiment_settings_dict):
    """

    @param experiment_settings_dict:
    @return: Transform to be applied to the dataset
    """
    partilal_aug_tmf =partial(aug_tfm,experiment_settings_dict=experiment_settings_dict)

    multi_tfm = RandTransform(enc=partilal_aug_tmf, p=1)
    return multi_tfm
'''

def get_camvid_dataset(experiment_settings_dict):
    """
    Train the publicly available CAMVID dataset
    from https://walkwithfastai.com/Segmentation
    https://github.com/walkwithfastai/walkwithfastai.github.io/blob/master/LICENSE
    """
    
    path = untar_data(URLs.CAMVID)
    valid_fnames = (path/'valid.txt').read_text().split('\n')
    get_msk = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}'
    codes = np.loadtxt(path/'codes.txt', dtype=str)

    def FileSplitter(fname):
        "Split `items` depending on the value of `mask`."
        valid = Path(fname).read_text().split('\n') 
        def _func(x): return x.name in valid
        def _inner(o, **kwargs): return FuncSplitter(_func)(o)
        return _inner
    name2id = {v:k for k,v in enumerate(codes)}
    void_code = name2id['Void']

    def get_some_files(path):
        return get_image_files(path)[-10:]

    if experiment_settings_dict["dev_mode"]:
        #When in dev_mode only use a few images
        get_image_files_function = get_some_files
    else:
        get_image_files_function = get_image_files



    def acc_camvid(inp, targ):
        targ = targ.squeeze(1)
        mask = targ != void_code
        return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()

    camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files_function,
                   splitter=RandomSplitter(),
                   get_y=get_msk,
                   batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)])
    dls = camvid.dataloaders(Path(experiment_settings_dict["path_to_images"]), bs=1)
    return dls


def get_dataset(experiment_settings_dict):
    """
    param: experiment_settings_dict: a dictionary holding all information nececeary to create a dataloader for a specific dataset
    return: a dataloader for the dataset

    if dev_mode == true dataset will be reduced to 20 images ,where 10 are used for validationset
    """
    #path to dataset in wich the 'images' folder is located
    pth= experiment_settings_dict["path_to_dataset"]
    

    if str(pth) =="camvid":
        return get_camvid_dataset(experiment_settings_dict)
    
    
    def develop_get_image_files(path,nr_of_train_images=20):
        """
        :param path:  NOT USED!
        :param nr_of_train_images: number of images to train on
        
        When developing we one want to train on a smaller portion of the images
        reads the all.txt file but returns a list with only the first items
        """
        all_files = (Path(experiment_settings_dict["path_to_all_txt"])).read_text().split('\n')
        
        return [Path(experiment_settings_dict["path_to_images"])/Path(a_path) for a_path in all_files[-nr_of_train_images:]] #get_image_files(path)[0:100] #when developing we only train on 100 of the images
    def sdfe_get_image_files(path):
        """
        param: path NOT USED!
        param: path NOT USED!

        reads in all.txt and returns a list with the paths to all images
        """
        all_files = (Path(experiment_settings_dict["path_to_all_txt"])).read_text().split('\n')
        #make sure that all files are of correct type

        im_type= experiment_settings_dict["im_type"]

        all_files=[im_file for im_file in all_files if im_type in im_file]
        return [Path(experiment_settings_dict["path_to_images"])/Path(a_path) for a_path in all_files]

    if experiment_settings_dict["dev_mode"]:
        sdfe_get_images = develop_get_image_files
        #when in dev mode we use a subset of the reduced trainingset for validation
        valid_fnames= [pathlibpath.name for pathlibpath in develop_get_image_files(pth) ][0:10]
        
    else:
        sdfe_get_images =sdfe_get_image_files
        #validationset
        #sdfe datasets are usually based on all.txt and valid.txt files
        if experiment_settings_dict["path_to_valid_txt"]:
            valid_fnames = (Path(experiment_settings_dict["path_to_valid_txt"])).read_text().split('\n')
        else:
            valid_fnames = None
    
    def ListSplitter(valid_items):
        """
        param: valid_items: a list with filenames that should be used for validation (and NOT for training)
        returns: a function that returns two masks, one filtering away all items used for validation, and one filtering away all items used for training
        """

        def _inner(items):
            
            val_mask = tensor([o.name in valid_items for o in items])
            return [~val_mask, val_mask]
        return _inner



    #make sure that all files are of correct type (we get problems if the .txt file includes e.g empty lines )
    label_type= experiment_settings_dict["label_image_type"]
    label_folder=Path(experiment_settings_dict["path_to_labels"])

    im_type= experiment_settings_dict["im_type"] 
    if experiment_settings_dict["path_to_valid_txt"]:
        
        valid_fnames=[im_file for im_file in valid_fnames if im_type in im_file]

        

        dataset_splitter= ListSplitter(valid_fnames)
    else:
        dataset_splitter= RandomSplitter()

    

   
    codes = np.loadtxt(experiment_settings_dict["path_to_codes"], dtype=str);
    print(" codes: "+str( codes))
    

    
    
    #IF folder structure is images/subfolder/image.tif
    #label_func = lambda o: pth / ('labels/masks/') / f'{o.stem}{o.suffix}'
    def label_func(image_pathlib_path):
        label_path =  label_folder / f'{image_pathlib_path.stem}{label_type}'
        return np.array(Image.open(label_path),dtype=np.int32)
    def building_func(image_pathlib_path):
        building_path =  Path(experiment_settings_dict["path_to_buildings"]) / f'{image_pathlib_path.stem}{label_type}'
        return np.array(Image.open(building_path))

    def label_plus_building_func(image_pathlib_path):
        """
        Use this function instead of the label_func() function when adding biulding footprints as a label

        update all pixles in the label to the building value IF they have a non-zero value in the bilding label.
        :param image_pathlib_path:
        :return:
        """
        label = label_func(image_pathlib_path)
        if not (Path(experiment_settings_dict["path_to_buildings"]) / f'{image_pathlib_path.stem}{label_type}').is_file():
            #print("building file is missing")
            return label
        else:
            pass #print("building file exist")

        building_label =building_func(image_pathlib_path)
        green_roof_mask= label==5
        building_mask= building_label!=0
        #set label-pixels to buliding value
        label[building_mask] = building_label[building_mask]
        #all green roof pixels have been overwritten to buldings. make them green roof again.
        label[green_roof_mask] = 5
        return label

    def get_label_func(experiment_settings_dict):
        if not "extra_labels" in experiment_settings_dict:
            return label_func
        if experiment_settings_dict["extra_labels"]=="buildings":
            input("extending labels with buildings")
            return label_plus_building_func
        elif experiment_settings_dict["extra_labels"]=="None":
            #input("using ordinary labels")
            return label_func
        else:
            print(experiment_settings_dict["extra_labels"])
            sys.exit("experiment_settings_dict['extra_labels'] should be 'buildings' or 'None'")
            






    if experiment_settings_dict["transforms"]:
        
        sdfi_tfm = sdfi_transforms.get_transforms(experiment_settings_dict = experiment_settings_dict)

        #sdfi_tfm = sdfi_transforms.get_transforms(["GaussNoise","Transpose","RandomBrightness","RandomShadow","ShiftScaleRotate"])
        

    else:
        sdfi_tfm=[]
    


    #The standard for fastai2 is to use Imageblocks but we get problems because of the extra channels and therfore use the underlaying DataBlock class instead
    #a_dataset = DataBlock(blocks= (TransformBlock(partial(open_img,experiment_settings_dict=experiment_settings_dict)), MaskBlock(codes)),get_items=sdfe_get_images, splitter=dataset_splitter,get_y= label_func ,item_tfms=sdfi_tfm,)
    def get_some_files(path):
        return get_image_files(path)[-10:]

    if experiment_settings_dict["dev_mode"]:
        #When in dev_mode only use a few images
        get_image_files_function = get_some_files
    else:
        get_image_files_function = get_image_files


    multichannel_stats= (experiment_settings_dict["means"],experiment_settings_dict["stds"])
    #nir_mean= 0.40779018 #based on a sample size of 1! OBS! UNKNOWN
    #nir_std=0.15176418
    #multichannel_stats = ([0.485, 0.456, 0.406,nir_mean], [0.229, 0.224, 0.225,nir_std])

    batch_tfms=[Normalize.from_stats(*multichannel_stats)]
    #This worked but I got problems with Normalize.encode(wants a tensorimage) (in site-packages/fastai/data/transforms.py) so we could replace normalize with NormalizeMultichannelImage

    
    a_dataset = DataBlock(blocks=(ImageBlockReplacement.MultichannelImageBlock(cls=ImageBlockReplacement.MultiChannelImage,experiment_settings_dict=experiment_settings_dict), MaskBlock(codes)),
                   get_items=sdfe_get_images,
                   splitter=dataset_splitter,
                   get_y=get_label_func(experiment_settings_dict),
                   item_tfms=sdfi_tfm,
                   batch_tfms=batch_tfms)  #*aug_transforms(),
                   
    """
    a_dataset = DataBlock(blocks=(ImageBlock(cls=ImageBlockReplacement.MultiChannelImage,experiment_settings_dict=experiment_settings_dict), MaskBlock(codes)),
                          get_items=get_image_files_function,
                          splitter=RandomSplitter(),
                          get_y=label_func,
                          item_tfms=sdfi_tfm,
                          batch_tfms=[Normalize.from_stats(*imagenet_stats)])  #*aug_transforms(),
    """
    
    #camvid.summary(source=Path(experiment_settings_dict["path_to_dataset"]) / "images")

    #Setting number of workers > 0 should speed up training if data is stored on slow medium lika NAS or HDD, to get this to work on windows is however tricky,se https://github.com/pytorch/pytorch/issues/16943
    if experiment_settings_dict.get("pin_memory"):
        print("using pin_memory ########################################################################")
    else:
        print("not using pin_memory #####################################################################")
    print("experiment_settings_dict['prefetch_factor']: "+str(experiment_settings_dict["prefetch_factor"]))

    dls = a_dataset.dataloaders(Path(experiment_settings_dict["path_to_images"]) , bs=experiment_settings_dict["batch_size"], num_workers=int(experiment_settings_dict["num_workers"]) ,pin_memory = experiment_settings_dict.get("pin_memory"),prefetch_factor = int(experiment_settings_dict["prefetch_factor"]))


    #if we use cropping as augmetnation for training we might want to reduce the batchsize during validation to make sure we dont use to much memory
    #during validation ,batchnorm layer use the running statistics collected during training so it should be safe to use batchsize=2 during validation
    # Customize DataLoaders to have different batch sizes
    #train_dl = dls.train.new(bs=experiment_settings_dict["batch_size"])
    #valid_dl = dls.valid.new(bs=2)

    # Create a new DataLoaders object with the custom DataLoader instances
    #dls = DataLoaders(train_dl, valid_dl)

    #Lastly let's make our vocabulary a part of our DataLoaders, as our loss function needs to deal with the Void label
    dls.vocab = codes
    
    return dls
