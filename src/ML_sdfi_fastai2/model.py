#!/usr/bin/env python
# coding: utf-8
"""
fastai2 kode for træning af semantic segmentation på vores dataset med >3 kanaler
en merge af kode fra
1 Baseret på https://walkwithfastai.com/Segmentation  (self_attention layer og flat_cos learningrate sceduler)
og
2. jupyter notebook Morten fand på internet  (hvordan man bruger sit eget dataset)
3. https://www.kaggle.com/cordmaur/remotesensing-fastai2-multiband-augmentations/notebook (hvordan man bruger flere chanels (OBS, augmetnations fra denne er helt forkerte!))
4.https://dpird-dma.github.io/blog/Multispectral-image-classification-Transfer-Learning/
"""

from fastcore.xtras import Path
import rasterio
import albumentations as A
from fastai.callback.hook import summary
from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import lr_find, fit_flat_cos
from fastai.callback.schedule import minimum, steep, slide, valley
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
from fastai.vision.learner import unet_learner
import torchvision.transforms as tfms
from PIL import Image
import numpy as np
import sys
from torch import nn
from torchvision.models.resnet import resnet34
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.vision import *
import time
import pathlib





print("segmentation4channelssimplified is UNDER DEVELOPMENT!")
print("data augmetnations NOT aplied since the proposed solution NOT works as intended")
print("images are normalised manualy with mean and std , NOT verified if this is done allready somewhere in the fastai pipeline")
print("waiting for 1 second")
time.sleep(1)

def plot_filters_single_channel(t):
    """
    arg t: tensor for a convlayer . e.g with shape t.shapetorch.Size([64, 4, 7, 7]) (64 kernels,4 channels, 7x7 pixelswide )
    """
    print(":t.shape"+str(t.shape))

    num_kernels = t.shape[0]
    num_channels = t.shape[1]
    receptive_field_height = t.shape[2]
    receptive_field_width = t.shape[3]
    
    
    #kernels depth * number of kernels
    nplots = num_kernels*num_channels
    ncols = 12
    
    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)
    
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    
    #looping through all the kernels in each channel
    for i_kernel in range(t.shape[0]):
        for j_channel in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i_kernel, j_channel].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i_kernel) + ',' + str(j_channel))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
   
    plt.tight_layout()
    plt.show()

def plot_filters_multi_channel(t):
    
    #get the number of kernals
    num_kernels = t.shape[0]    
    
    #define number of columns for subplots
    num_cols = 12
    #rows = num of kernels
    num_rows = num_kernels
    
    #set the figure size
    fig = plt.figure(figsize=(num_cols,num_rows))
    
    #looping through all the kernels
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        
        #for each kernel, we convert the tensor to numpy 
        npimg = np.array(t[i].numpy(), np.float32)
        #standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
    plt.savefig('myimage.png', dpi=100)    
    plt.tight_layout()
    plt.show()

def plot_weights(model,  single_channel = True, collated = False):
    

    """
    :param model: a nn.module object (e.g a conv layer)
    :param single_channel: shouldwe show each channel as a separate image

    copied from https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
    only meant to work for lenet5 on 3channel images but with single_channel=true it still should work for any number of channels
    slightly modified the code to run on resnets instead of lenet5


    """

  
    #extracting the model features at the particular layer number
    layer = model

    #checking whether the layer is convolution layer or not 
    if isinstance(layer, nn.Conv2d):
        #getting the weight tensor data
        weight_tensor = layer.weight.data

        if single_channel:
            if collated:
                plot_filters_single_channel_big(weight_tensor)
            else:
                plot_filters_single_channel(weight_tensor)
            
        else:
            if weight_tensor.shape[1] == 3:
                plot_filters_multi_channel(weight_tensor)
            else:
                print("Can only plot weights with three channels with single channel = False")
            
    else:
        print("Can not visualize")
        print(layer)
        print("Can only visualize layers which are convolutional")
        


def get_channel(channel,extra_channel_shape ):
    """
    :param channel: the kind of channel to ad e.g "ni" (near infra-read) or "d" (depth/height)
    :param extra_channel_shape: the shape of the channel to add
    :return: the channel as numpy array
    """
    use_dummy_values= True
    if use_dummy_values:
        new_channel=np.zeros(extra_channel_shape)
    else:
        sys.exit("TO DO! either read the info directly from the image or from somewhere else..")
        new_channel = "TODO"
    return new_channel


# open an image and convert it to a tensor
def open_img(path,channels=['r','g','b','ni'],channels_to_zero_out=[]):
    """
    :param path: path to image 
    :param resize:
    :param channels: list of channels to load from image
    :channels_to_zero_out: list of  indexes to channels that will be multiplied with 0 (used to verify that model use the values in the channelæ)
    :return:

    images should be normalizes like this
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683/19
    https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
    output[channel] = (input[channel] - mean[channel]) / std[channel]
    """
    #     use rasterio to open image as numpy array and rescale to 0-1
    #     you may need to change this if you have values above 255
    img_as_array = rasterio.open(path).read().astype('float32')# looks like division by 255 is done in batch_tfms allready but I dont see how! /255.0 (most probably by the astype('float32'))
    nir_channel= img_as_array[-1]

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
        mean_and_std_normalized=(img_as_array.transpose()-np.array([0.485, 0.456, 0.406,nir_mean],dtype=np.float32)/np.array([0.229, 0.224, 0.225,nir_std],dtype=np.float32) ).transpose()
        img_as_array=mean_and_std_normalized
    #print("mean_and_std_normalized : "+str(mean_and_std_normalized ))


    #input(np.array(img_as_array).mean())

    
    
    #input(img_as_array.mean())


    #no need to ad nir-channel since it gets loaded automatically by rasterio.open
    #We can use the following procedure for other kinds of channels, e.g lidar 
    ad_extra_channel= False 
    if ad_extra_channel:
        extra_channels= list(set(channels) -set(["r","g","b","ni"]))
        for channel in extra_channels:
            extra_channel_shape= [1]+list(img_as_array.shape[1:])
            new_channel=get_channel(channel,extra_channel_shape = extra_channel_shape)
            img_as_array=np.array(np.vstack((img_as_array,new_channel)),dtype=np.float32)

    #verify that we now have the correct number of channels
    assert len(channels) ==img_as_array.shape[0]

    #     convert numpy array to tensor
    img_as_tensor = torch.from_numpy(img_as_array)
    if len(channels_to_zero_out)>0:
        for channel_to_zero in channels_to_zero_out:
            img_as_tensor[channel_to_zero]=0
            
    

    

    return img_as_tensor

# resize the dimensions of a tensor
def resize_tensor(input_tensor,img_size):
    """
    :param input_tensor:  image as a tensor
    :param img_size new size of image
    :return:

    Not used!

    """
    #     from https://stackoverflow.com/questions/59803041/resize-rgb-tensor-pytorch
    tensor_un = input_tensor.unsqueeze(0)
    tensor_res = torch.nn.functional.interpolate(tensor_un,size=(img_size,img_size), mode='bilinear', align_corners=True)
    tensor_sq = tensor_res.squeeze(0)
    return(tensor_sq)



# we can't use the built in fastai augmentations as they expect 3 channel images so we are using Albumentations instead
# https://github.com/albumentations-team/albumentations
# add as many as you want but these are executed on CPU so can be slow...
transform_list = [A.RandomBrightnessContrast(p=1,brightness_limit=.2),
                  A.RandomRotate90(p=0.5),
                  A.HorizontalFlip(p=.5),
                  A.VerticalFlip(p=.5),
                  A.Blur(p=.1),
                  A.Rotate(p=0.5,limit = 10)
                  ]

# apply the augmentations in a loop
def aug_tfm(tensor):
    """
    :param tensor: image as a tensor
    :return: transformed version of tensor

    Data augmentation during training

    """
    #     this function is used for both images and labels so check the count of the input tensor
    #     if the count is above 1 its not a label so apply the augmentations
    #print("label and data both goa through transformation")
    #print(type(tensor))


    #since sdfes lables not are tensors but fastai fastai.vision.core.PILMask ,we onoy apply the transforms if type is torch.Tensor
    #OBS! if we want to aply rotations both labels and images need to be transformed!!

    use_augmentation = False # The flips and roations must be done on the labels also! but brightnes and such should only be done on image. the solution below can not work properly!

    if use_augmentation and type(tensor) == torch.Tensor:#tensor.nelement()>1:
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

    return tensor

multi_tfm = RandTransform(enc=aug_tfm, p=1)







def get_dataset(experiment_settings_dict):
    """
    param: experiment_settings_dict: a dictionary holding all information nececeary to create a dataloader for a specific dataset
    return: a dataloader for the dataset
    """
    #path to dataset in wich the 'images' folder is located
    pth= experiment_settings_dict["path_to_dataset"]
    
    def develop_get_image_files(path,nr_of_train_images=20):
        """
        :param path:  NOT USED!
        :param nr_of_train_images: number of images to train on
        
        When developing we one want to train on a smaller portion of the images
        reads the all.txt file but returns a list with only the first items
        """
        all_files = (Path(experiment_settings_dict["path_to_all_txt"])).read_text().split('\n')
        
        return [pth/Path("images")/Path(a_path) for a_path in all_files[0:nr_of_train_images]] #get_image_files(path)[0:100] #when developing we only train on 100 of the images
    def sdfe_get_image_files(path):
        """
        param: path NOT USED!

        reads in all.txt and returns a list with the paths to all images
        """
        all_files = (Path(experiment_settings_dict["path_to_all_txt"])).read_text().split('\n')
        #make sure that all files are of correct type
        im_type= ".tif"
        all_files=[im_file for im_file in all_files if im_type in im_file]
        return [pth/Path("images")/Path(a_path) for a_path in all_files] 

    if experiment_settings_dict["dev_mode"]:
        sdfe_get_images = develop_get_image_files
        #when in dev mode we use a subset of the reduced trainingset for validation,train and validate on same images
        valid_fnames= [pathlibpath.name for pathlibpath in develop_get_image_files(pth) ][0:10]
        textfile = open("./dev_mode_valid_images.txt", "w")
        for element in valid_fnames:
            textfile.write(element + "\n")
        textfile.close()
    else:
        sdfe_get_images =sdfe_get_image_files
        #validationset
        #sdfe datasets are based on all.txt and valid.txt files
        valid_fnames = (Path(experiment_settings_dict["path_to_valid_txt"])).read_text().split('\n')

    #make sure that all files are of correct type (we get problems if the .txt file includes e.g empty lines )
    im_type= ".tif"
    valid_fnames=[im_file for im_file in valid_fnames if im_type in im_file]
    

   
    codes = np.loadtxt(experiment_settings_dict["path_to_codes"], dtype=str);
    print(" codes: "+str( codes))
    

    def ListSplitter(valid_items):
        """
        param: valid_items: a list with filenames that should be used for validation (and NOT for training)
        returns: a function that returns two masks, one filtering away all items used for validation, and one filtering away all items used for training
        """
        def _inner(items):
            val_mask = tensor([o.name in valid_items for o in items])
            return [~val_mask, val_mask]
        return _inner

    def extract_subfolder(path):
        """
        param  path: path to imagefile
        return: folders name
        
        If the image is in a subfolder we need to know the name so we can use it for finding the label
        """
        
        return Path(path).parents[0].name

    #IF folder structure is images/subfolder/image.tif
    #label_func = lambda o: pth / ('labels/masks/' + extract_subfolder(o)) / f'{o.stem}{o.suffix}'
    #IF folder structure is images/subfolder/image.tif
    label_func = lambda o: pth / ('labels/masks/') / f'{o.stem}{o.suffix}'


    #The standard for fastai2 is to use Imageblocks but we get problems because of the extra channels and therfore use the underlaying DataBlock class instead
    camvid = DataBlock(blocks= (TransformBlock(partial(open_img,channels=experiment_settings_dict["channels"])), MaskBlock(codes)),get_items=sdfe_get_images, splitter=ListSplitter(valid_fnames),get_y= label_func ,item_tfms=multi_tfm,)
    print("################################################################################################################")
    print("OBS! batch_tfms should be added again!, with *imagenet_stats replaced with the std and avg values from rgbnir, item_tfms should be removed as is done in my camvid_og_SDFE_data.ipynb")
    # batch_tfms=[*aug_transforms(size=(360, 480)), Normalize.from_stats(*imagenet_stats)]  #(transforms must be adapted to 4 channels) OBS! I SHOULD BE ABLE TO INPUT MY STD and MEAN VALUES HERE!
    print("################################################################################################################")
    camvid.summary(source=Path(experiment_settings_dict["path_to_dataset"]) / "images")

    dls = camvid.dataloaders(Path(experiment_settings_dict["path_to_dataset"]) / 'images', bs=experiment_settings_dict["batch_size"], num_workers=0)
    #Lastly let's make our vocabulary a part of our DataLoaders, as our loss function needs to deal with the Void label
    dls.vocab = codes
    return dls

class basic_traininFastai2:
    def __init__(self,experiment_settings_dict,dls):
        """
        :param experiment_settings_dict: a dictionary holding the parameters for the training
        :param dls: a dataloader feed to the unet learner
        """
        self.experiment_settings_dict =experiment_settings_dict
        self.learn = self.get_basic_training(experiment_settings_dict,dls)

    def find_learning_rate(self,show_images):
        """
        :param show_images: boolen that decides if the plot should be shown to the user
        :return: the Lr suitable for model and dataset (OBS. should be multiplied with e.g 15 in order to fit with fit_one_cykle )
        """
        lrs = self.learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
        lr_min, lr_steep, lr_slide, lr_valley = self.learn.lr_find(suggest_funcs=(minimum, steep, slide, valley))
        print("lr_min:"+ str(lr_min))
        print("lr_steep:" + str(lr_steep))
        print("lr_slide:" + str(lr_slide))
        print("lr_valley:" + str(lr_valley)) #recomended in https://forums.fast.ai/t/new-lr-finder-output/89236/3
        if show_images:
            print("exit the graph to continue")
            plt.show()
        print("use lr_valley as learningrate:" + str(lr_valley))
        lr_max = lr_valley
        return lr_max



    def load_learn_cycle(self,lr_max):
        """
        :param lr_max: the max learningrate that the fit_one_cyckle will cykle towards and back from
        :return: None

        Trains a model on a dataset using fit_one_cykle
        the 'load' in load_learn_cycle is legacy and meant to be compatible with sdfe's older fastai1 code
        """
        if self.experiment_settings_dict["load_newest"]:
            sys.exit("CHECK HOW THIS IS HANDEDLED IN sdfe.basictraining !")
            self.learn.load(job_name)

        if self.experiment_settings_dict["freeze"]:
            self.learn.freeze()
        else:
            self.learn.unfreeze()
        #IN order to monmitor the acuracy we might have to modifie the learner as they talk about here https://forums.fast.ai/t/equivalent-of-add-metrics-in-fastai2/77575/2
        self.learn.fit_one_cycle(n_epoch=self.experiment_settings_dict["epochs"], lr_max=lr_max, cbs=[SaveModelCallback(every_epoch= True, monitor='valid_loss', fname=self.experiment_settings_dict["job_name"]),CSVLogger(fname= self.experiment_settings_dict["job_name"]+".csv", append=True)])

        #get_c(dls)

        print("saving model")
        self.learn.save(self.experiment_settings_dict["job_name"])




    def get_basic_training(self,experiment_settings_dict,dls,ignore_index=-100):
        """
        :param experiment_settings_dict: a dictionary holding the parameters for the training
        :param dls: a dataloader feed to the unet learner
        :return: a unet learner that use data from the dataloader
        """
        #make sure the path/to/folder that the model should be saved in exists
        pathlib.Path(self.experiment_settings_dict["model_folder"]).mkdir(parents=True, exist_ok=True)
        #make sure the path/to/folder that the model should be saved in exists
        pathlib.Path(self.experiment_settings_dict["log_folder"]).mkdir(parents=True, exist_ok=True)
        
        
        def valid_accuracy(inp, targ):
            """
            valid_accuracy runs after epoch, calculating the accuracy on the validationset
            Dokumentation that shows that the metrics per default is computed only on the validationset can be ound here: https://docs.fast.ai/learner.html#Learner
            """
            targ = targ.squeeze(1)
            
            void_code=-1 #use a code that does not exist in the dataset
            #the masked target is the same as the target (we dont use 'dont care labels')
            mask = targ != void_code  #Rasmus is pretty sure voide code is the "do not care label (e.g should not influence the cost function)"
            
            return (inp.argmax(dim=1)[mask] == targ[mask]).float().mean()

        if experiment_settings_dict["sceduler"]== "fit_one_cycle":
            
            #the default loss function for a fastai2 unet is a flattened CrossEntropyloss. We here define it ourselfs so we can modify the ignore_index (what label should be ignored when computing the loss)
            #https://forums.fast.ai/t/loss-function-of-unet-learner-flattenedloss-of-crossentropyloss/51605
            #https://docs.fast.ai/losses.html#CrossEntropyLossFlat
            
            a_loss_func= CrossEntropyLossFlat(axis=1,ignore_index=ignore_index) #reduction decisdes if the output should be summed or averaged in any way (this changes teh dimensions of the loss)
            learn = unet_learner(dls, self.experiment_settings_dict["model"], loss_func=a_loss_func,metrics=valid_accuracy, wd=1e-2,
                                 path= self.experiment_settings_dict["log_folder"],
                                 model_dir=self.experiment_settings_dict["model_folder"] ,n_in=len(experiment_settings_dict["channels"]))#callback_fns=[partial(CSVLogger, filename= experiment_settings_dict["job_name"], append=True)])
        else:
            sys.exit("in order to use other learningrate schedule than fit_one_cykle  copy functionality from segmentation4channels.py")

        return learn




def train(experiment_settings_dict):
    """
    :param experiment_settings_dict: a dictionary holding the parameters for the training
    :return: None

    1.loads a dataset
    2.configures a unet-training
    3.sets a learningrate 
    4.trains
    """
    run_on_cpu=False
    if run_on_cpu:
        defaults.device = torch.device('cpu')

    print("loading the dataset..")
    
    dls = get_dataset(experiment_settings_dict)
    print("setting up a unet training..")

    training= basic_traininFastai2(experiment_settings_dict,dls)

    if "lr" in experiment_settings_dict:
        max_lr = experiment_settings_dict["lr"]
        print("using predefined max learning rate :"+str(max_lr))
        
    else:
        print("Finding learning rate...")
        max_lr= training.find_learning_rate(show_images=False)
    print("max_lr:"+str(max_lr))

    print("run the training..")
    print("job_name: " + experiment_settings_dict["job_name"])

    training.load_learn_cycle(lr_max=max_lr)

    print("TRAINING DONE! job_name: " + experiment_settings_dict["job_name"])

def visualize_model(experiment_settings_dict):
    #create a classifier
    dls = get_dataset(experiment_settings_dict)
    training= basic_traininFastai2(experiment_settings_dict,dls)
    if experiment_settings_dict["load_newest"]:
        #load save weights
        training.learn.load(experiment_settings_dict["job_name"])

    print("#The model graph##")
    print(training.learn.model)
    print("##################")
    print("The first layer")
    the_first_Sequential = training.learn.model.layers[0]
    print(the_first_Sequential)
    the_first_layer= the_first_Sequential[0]
    print(the_first_layer)
    
    print("##################")
    print(the_first_layer.weight)
    print(the_first_layer.weight.shape)
    print("################STATE DICT of complete modekl#########")
    print(training.learn.model.state_dict())
    print("################STATE DICT  of first layer#########")
    print(the_first_layer.state_dict())

    print("visualizing weights as images")
    plot_weights(model=the_first_layer, single_channel = True, collated = False)


    """
    from torchviz import make_dot
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    """

    
def classify_all(experiment_settings_dict,benchmark_folder,output_folder,develop,show,all_txt):
    """
    :param experiment_settings_dict: a dictionary holding the parameters for the trainer that will be used for classification
    :param benchmark_folder: the folder with images that will be clæassified 
    :param output_folder: the folder where the resuling predictions will be saved
    :param: develop : when developing we validate on the images used for validation during training OBS only works if benchmark_folder == training_dataset_folder
    :param: show: should the predictions be visualized after classification?
    :return: None

    Saves predictions in output_folder
    """
    print("CLASSIFY ALL STARTS.")
    #list paths to all images   
    all_files = Path(all_txt).read_text().split('\n')
    all_files=[Path(benchmark_folder)/Path("images")/Path(a_path) for a_path in all_files]
    #all_files = [experiment_settings_dict["path_to_dataset"]/Path("images")/Path("")/Path(a_path) for a_path in all_files]
    if develop:
        use_develop_txt = True #classify the images used to validate the training done in develop_mode
        if use_develop_txt:
            all_files = Path("./dev_mode_valid_images.txt").read_text().split('\n')
            all_files=[Path(benchmark_folder)/Path("images")/Path("83_25_26")/Path(a_path) for a_path in all_files]
        else:
            all_files=all_files[:10]
    

    #make sure that all files are of correct type
    im_type= ".tif"
    all_files=[im_file for im_file in all_files if im_type in im_file.name]

    #create a classifier
    dls = get_dataset(experiment_settings_dict)
    training= basic_traininFastai2(experiment_settings_dict,dls)
    #load save weights
    training.learn.load(experiment_settings_dict["job_name"])

    #make sure outputfolder exists
    os.makedirs(output_folder, exist_ok=True)



    #classify all images in benchmark_folder
    for a_file in all_files:
        print("classifying : "+str(a_file),end='\r')
        
        dl = training.learn.dls.test_dl([a_file]) # dl = training.learn.dls.test_dl(all_files) # 
        #Does not work with 4 chanel tensors
        #if show:
        #    dl.show_batch()

        
        preds = training.learn.get_preds(dl=dl)
        pred_1 = preds[0][0]
        pred_arx = pred_1.argmax(dim=0)

        if show:
            plt.imshow(pred_arx)
            plt.show()
        
        Image.fromarray(np.array(pred_arx,dtype=np.uint8)).save(Path(output_folder)/Path(a_file.name))
    print("CLASSIFY ALL DONE, result in : "+str(output_folder))








if __name__ == "__main__":
    """
    Given one or more dictionaries that defines a training
    1. Train a model
    2. Classify images
    """
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    experiment_settings_fit_flat_cos ={"path_to_dataset": Path(r"\mnt\trainingdata\rooftop_1000p"),
                                       "load_newest":False,
                                       "train":True,
                                       "predict":False,
                                       "sceduler":"fit_flat_cos", #'fit_flat_cos'
                                       "dev_mode":True,
                                       "path_to_valid_txt": r"\mnt\trainingdata\rooftop_1000p\valid_esbjergplusen_no_subfolders_in_name.txt",
                                       "epochs":9,
                                       "frozenepochs":1,
                                       "job_name": "fastai2experiment_fit_flat_cos",
                                       "channels":['r','g','b']
                                       }

    experiment_settings_CAMVID = {"path_to_dataset": untar_data(URLs.CAMVID),
                                  "load_newest": False,
                                  "train": True,
                                  "predict":False,
                                  "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                                  "dev_mode": False,
                                  "path_to_valid_txt": untar_data(URLs.CAMVID) / 'valid.txt',
                                  "epochs": 10,
                                  "frozenepochs": 1,
                                  "job_name": "fastai2experiment_CAMVID",
                                  "channels":['r','g','b']
                                  }
    multichannels_linux = {"path_to_dataset": Path(r"/mnt/trainingdata-disk/trainingdata/rooftop_1000p"),
                     "load_newest": False,
                     "train": True,
                     "predict":False,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": True,
                     "path_to_valid_txt": r"/mnt/trainingdata-disk/trainingdata/rooftop_1000p/valid_esbjergplusen_no_subfolders_in_name.txt",
                     "epochs": 10,
                     "freeze":False,
                     "job_name": "multichannels",
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/trainingdata-disk/fastai2logs",
                     "model_folder":"/mnt/trainingdata-disk/fastai2models",
                     "batch_size":2

                     }
    # A dictionary defining a fast testtraining
    dev_mode_multichannels_windows = {"path_to_dataset": Path(r"\mnt\trainingdata\landsddaekkende_dataset\testset"),
                     "load_newest": False,
                     "train": False,
                     "predict":True,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"\mnt\trainingdata\landsddaekkende_dataset\testset\valid_no_black_images.txt",
                     "epochs": 10,
                     "freeze":False,
                     "job_name": "devmode_multichannels_resnet50_landsdakendetestset",
                     "model":resnet50,
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs/devmode",
                     "model_folder":"/mnt/models/fastai2models/devmodes",
                     "batch_size":2,
                     "lr":3.0199516913853586e-05, #remove this to use learningrate finder instead
                     "path_to_all_txt":r"/mnt/trainingdata/landsddaekkende_dataset/testset/all_no_black_images.txt",
                     "benchmark_folder": r"\mnt\trainingdata\landsddaekkende_dataset\testset" # r"/mnt/trainingdata/Vector2ImageEsbjergplus" #r"\mnt\trainingdata\AmagerLangelandRaabilleder" #r"/mnt/trainingdata/Vector2ImageEsbjergplus" #
                     }
    

    best_lr=0.0009059855074156076 # when using range_finer valley and multiplying result with 30 I got the best result with net : valid loss 0.04206996411085129
    multichannels_windows = {"path_to_dataset": Path(r"/mnt/trainingdata/Vector2ImageEsbjergplus"),
                     "load_newest": False,
                     "train": True,
                     "predict": True,
                     "predict":False,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"/mnt/trainingdata/Vector2ImageEsbjergplus/persesbjergplus_valid.txt",
                     "epochs": 20,
                     "freeze":False,
                     "job_name": "multichannels_exp2",
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs",
                     "model_folder":"/mnt/models/fastai2models",
                     "batch_size":4,
                     "lr":3.0199516913853586e-05, #remove this to use learningrate finder instead
                     "path_to_all_txt":r"/mnt/trainingdata/Vector2ImageEsbjergplus/persesbjergpluslimited_all.txt",
                     "benchmark_folder": r"\mnt\trainingdata\AmagerLangelandRaabilleder" # r"/mnt/trainingdata/Vector2ImageEsbjergplus" #r"\mnt\trainingdata\AmagerLangelandRaabilleder" #r"/mnt/trainingdata/Vector2ImageEsbjergplus" #
                     }


    #resnet18, resnet34, resnet50, resnet101, resnet152
    resnet152_config = {"path_to_dataset": Path(r"/mnt/trainingdata/Vector2ImageEsbjergplus"),
                     "load_newest": True,
                     "train": True,
                     "predict":True,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"/mnt/trainingdata/Vector2ImageEsbjergplus/valid.txt",
                     "epochs": 20,
                     "freeze":False,
                     "job_name": "resnet152_20epochs_secondtry",
                     "model":resnet152,
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs",
                     "model_folder":"/mnt/models/fastai2models",
                     "batch_size":2,
                     "lr":best_lr, 
                     "path_to_all_txt":r"/mnt/trainingdata/Vector2ImageEsbjergplus/all.txt",
                     "benchmark_folder": r"/mnt/trainingdata/Vector2ImageEsbjergplus"
                     }
    resnet101_config = {"path_to_dataset": Path(r"/mnt/trainingdata/Vector2ImageEsbjergplus"),
                     "load_newest": False,
                     "train": True,
                     "predict":False,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"/mnt/trainingdata/Vector2ImageEsbjergplus/persesbjergplus_valid.txt",
                     "epochs": 20,
                     "freeze":False,
                     "job_name": "resnet101_20epochs",
                     "model":resnet101,
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs",
                     "model_folder":"/mnt/models/fastai2models",
                     "batch_size":2,
                     "lr":best_lr, 
                     "path_to_all_txt":r"/mnt/trainingdata/Vector2ImageEsbjergplus/persesbjergpluslimited_all.txt",
                     "benchmark_folder": r"\mnt\trainingdata\AmagerLangelandRaabilleder" # r"/mnt/trainingdata/Vector2ImageEsbjergplus" #r"\mnt\trainingdata\AmagerLangelandRaabilleder" #r"/mnt/trainingdata/Vector2ImageEsbjergplus" #
                     }
    resnet50_config = {"path_to_dataset": Path(r"/mnt/trainingdata/Vector2ImageEsbjergplus"),
                     "load_newest": False,
                     "train": True,
                     "predict":False,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"/mnt/trainingdata/Vector2ImageEsbjergplus/valid.txt",
                     "epochs": 20,
                     "freeze":False,
                     "job_name": "resnet50_20epochs_Lossexperiments1",
                     "model":resnet50,
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs",
                     "model_folder":"/mnt/models/fastai2models",
                     "batch_size":2,
                     "lr":best_lr, 
                     "path_to_all_txt":r"/mnt/trainingdata/Vector2ImageEsbjergplus/all.txt",
                     "benchmark_folder": r"\mnt\trainingdata\AmagerLangelandRaabilleder" # r"/mnt/trainingdata/Vector2ImageEsbjergplus" #r"\mnt\trainingdata\AmagerLangelandRaabilleder" #r"/mnt/trainingdata/Vector2ImageEsbjergplus" #
                     }
    resnet34_config = {"path_to_dataset": Path(r"/mnt/trainingdata/Vector2ImageEsbjergplus"),
                     "load_newest": False,
                     "train": True,
                     "predict":False,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"/mnt/trainingdata/Vector2ImageEsbjergplus/valid.txt",
                     "epochs": 20,
                     "freeze":False,
                     "job_name": "resnet34_20epochs",
                     "model":resnet34,
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs/resnet34_config",
                     "model_folder":"/mnt/models/fastai2models/resnet34_config",
                     "batch_size":2,
                     "lr":best_lr, 
                     "path_to_all_txt":r"/mnt/trainingdata/Vector2ImageEsbjergplus/all.txt",
                     "benchmark_folder": r"\mnt\trainingdata\AmagerLangelandRaabilleder" # r"/mnt/trainingdata/Vector2ImageEsbjergplus" #r"\mnt\trainingdata\AmagerLangelandRaabilleder" #r"/mnt/trainingdata/Vector2ImageEsbjergplus" #
                     }
    resnet18_config = {"path_to_dataset": Path(r"/mnt/trainingdata/Vector2ImageEsbjergplus"),
                     "load_newest": False,
                     "train": True,
                     "predict":False,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"/mnt/trainingdata/Vector2ImageEsbjergplus/valid.txt",
                     "epochs": 20,
                     "freeze":False,
                     "job_name": "resnet18_20epochs",
                     "model":resnet18,
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs/resnet18_config",
                     "model_folder":"/mnt/models/fastai2models/resnet18_config",
                     "batch_size":2,
                     "lr":best_lr, 
                     "path_to_all_txt":r"/mnt/trainingdata/Vector2ImageEsbjergplus/all.txt",
                     "benchmark_folder": r"\mnt\trainingdata\AmagerLangelandRaabilleder" # r"/mnt/trainingdata/Vector2ImageEsbjergplus" #r"\mnt\trainingdata\AmagerLangelandRaabilleder" #r"/mnt/trainingdata/Vector2ImageEsbjergplus" #
                     }
    infer_with_old_model_resnet152_config_bykerne = {"path_to_dataset": Path(r"/mnt/trainingdata/Vector2ImageEsbjergplus"),
                     "load_newest": True,
                     "train": False,
                     "predict":True,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"/mnt/trainingdata/Vector2ImageEsbjergplus/valid.txt",
                     "epochs": 20,
                     "freeze":False,
                     "job_name": "resnet152_20epochs",
                     "model":resnet152,
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs",
                     "model_folder":"/mnt/models/fastai2models",
                     "batch_size":2,
                     "lr":best_lr, 
                     "path_to_all_txt":r"/mnt/trainingdata/Vector2ImageEsbjergplus/all.txt",
                     "benchmark_folder":r"F:\FRI\MEDARBEJDERE\PEMAC\VITrans\data\MLUDV199\src\bykerne"
                     }
    infer_with_old_model_resnet152_config_forstad = {"path_to_dataset": Path(r"/mnt/trainingdata/Vector2ImageEsbjergplus"),
                     "load_newest": True,
                     "train": False,
                     "predict":True,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"/mnt/trainingdata/Vector2ImageEsbjergplus/persesbjergplus_valid.txt",
                     "epochs": 20,
                     "freeze":False,
                     "job_name": "resnet152_20epochs",
                     "model":resnet152,
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs",
                     "model_folder":"/mnt/models/fastai2models",
                     "batch_size":2,
                     "lr":best_lr, 
                     "path_to_all_txt":r"/mnt/trainingdata/Vector2ImageEsbjergplus/persesbjergpluslimited_all.txt",
                     "benchmark_folder":r"F:\FRI\MEDARBEJDERE\PEMAC\VITrans\data\MLUDV199\src\forstad"
                     }
    infer_with_old_model_resnet152_config_land = {"path_to_dataset": Path(r"/mnt/trainingdata/Vector2ImageEsbjergplus"),
                     "load_newest": True,
                     "train": False,
                     "predict":True,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"/mnt/trainingdata/Vector2ImageEsbjergplus/persesbjergplus_valid.txt",
                     "epochs": 20,
                     "freeze":False,
                     "job_name": "resnet152_20epochs",
                     "model":resnet152,
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs",
                     "model_folder":"/mnt/models/fastai2models",
                     "batch_size":2,
                     "lr":best_lr, 
                     "path_to_all_txt":r"/mnt/trainingdata/Vector2ImageEsbjergplus/persesbjergpluslimited_all.txt",
                     "benchmark_folder":r"F:\FRI\MEDARBEJDERE\PEMAC\VITrans\data\MLUDV199\src\land"
                     }
    infer_with_old_model_resnet152_config_all = {"path_to_dataset": Path(r"/mnt/trainingdata/Vector2ImageEsbjergplus"),
                     "load_newest": False,
                     "train": False,
                     "predict":True,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"/mnt/trainingdata/Vector2ImageEsbjergplus/valid.txt",
                     "epochs": 20,
                     "freeze":False,
                     "job_name": "resnet152_20epochs",
                     "model":resnet152,
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs",
                     "model_folder":"/mnt/models/fastai2models",
                     "batch_size":2,
                     "lr":best_lr, 
                     "path_to_all_txt":r"/mnt/trainingdata/Vector2ImageEsbjergplus/all.txt",
                     "benchmark_folder":r"/mnt/trainingdata/AmagerLangelandRaabilleder"
                     }

    # A dictionary defining trainign on landsdekkende test
    resnet50_landsdakendetestset = {"path_to_dataset": Path(r"\mnt\trainingdata\landsddaekkende_dataset\testset"),
                     "load_newest": False,
                     "train": True,
                     "predict":True,
                     "sceduler": "fit_one_cycle",  # 'fit_flat_cos'
                     "dev_mode": False,
                     "path_to_valid_txt": r"\mnt\trainingdata\landsddaekkende_dataset\testset\valid_no_black_images.txt",
                     "epochs": 10,
                     "freeze":False,
                     "job_name": "resnet50_landsdakendetestset",
                     "model":resnet50,
                     "channels":['r','g','b','ni'],
                     "log_folder":"/mnt/logs/fastai2logs/devmode",
                     "model_folder":"/mnt/models/fastai2models/devmodes",
                     "batch_size":2,
                     "lr":3.0199516913853586e-05, #remove this to use learningrate finder instead
                     "path_to_all_txt":r"/mnt/trainingdata/landsddaekkende_dataset/testset/all_no_black_images.txt",
                     "benchmark_folder": r"\mnt\trainingdata\landsddaekkende_dataset\testset" # r"/mnt/trainingdata/Vector2ImageEsbjergplus" #r"\mnt\trainingdata\AmagerLangelandRaabilleder" #r"/mnt/trainingdata/Vector2ImageEsbjergplus" #
                     }
    
    """
    do_simple_lr_search =False
    if do_simple_lr_search:
        job_name= multichannels_windows["job_name"]
        for a_lr_muliplier in [20,15,30,5,40,80]:
            multichannels_windows["lr"]=a_lr_muliplier*3.0199516913853586e-05
            multichannels_windows["job_name"]=job_name+"muliplier"+str(a_lr_muliplier)
            train(experiment_settings_dict=multichannels_windows)
    """

    """
    do_longer_training = True
    if do_longer_training:
        multichannels_windows["lr"]=best_lr
        for nr_of_epochs in [20,50,100]:
            multichannels_windows["epochs"] = nr_of_epochs
            multichannels_windows["job_name"] = str(nr_of_epochs)+"epokswindows"
            train(experiment_settings_dict=multichannels_windows)
    """


    #finished with infer_with_old_model_resnet152_config_bykerne
     



    for experiment_settings_dict in [resnet50_landsdakendetestset]:#[dev_mode_multichannels_windows]:#[dev_mode_multichannels_windows,resnet18_config,resnet34_config]:#[resnet50_config]:#[infer_with_old_model_resnet152_config_all]:#[infer_with_old_model_resnet152_config_all]:
        #training
        do_visualize_model = False
        show = False
        if do_visualize_model:
            visualize_model(experiment_settings_dict=experiment_settings_dict)
            sys.exit("avaid trainign if visualizing weights")
            

        
        
        if experiment_settings_dict["train"]:
            train(experiment_settings_dict=experiment_settings_dict)
        
        if experiment_settings_dict["predict"]:
            #classification
            benchmark_folder = experiment_settings_dict["benchmark_folder"]
            output_folder = experiment_settings_dict["model_folder"]+pathlib.Path(benchmark_folder).name
            classify_all(experiment_settings_dict=experiment_settings_dict,benchmark_folder = benchmark_folder,output_folder=output_folder,develop=experiment_settings_dict["dev_mode"],show=show,all_txt=experiment_settings_dict["path_to_all_txt"]) 
        #if experiment_settings_dict["make_training_rapport"]:
            
        
        
        