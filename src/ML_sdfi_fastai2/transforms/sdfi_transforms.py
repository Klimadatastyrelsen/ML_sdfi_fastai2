#based on
#https://docs.fast.ai/10b_tutorial.albumentations.html
#if we want more cahnnels than 4 we need to do something like described here https://albumentations.ai/docs/examples/example_multi_target/

#mamba install -c conda-forge imgaug
#mamba install -c conda-forge albumentations

import os
import sys
import inspect
#we want acces to the libraries one directory above
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import albumentations
from albumentations import ShiftScaleRotate, RandomShadow
#from fastai.data.external import untar_data, URLs
#from fastai.data.transforms import get_image_files
from fastai.vision.core import PILImage, PILMask
#import matplotlib.pyplot as plt
from fastai.vision.all import *
import sdfi_dataset
import utils.utils as sdfi_utils
import argparse
from PIL import Image
from fastai.vision.data import imagenet_stats
import train
import infer
import pathlib
import ImageBlockReplacement
import matplotlib.cm as cm
import matplotlib.colors as mcolors


"""
THIS CLASS IS REPLACED WIHT THE VERSION IN ImageBLokREplacment
#Defining a class hat is identicall to a torch Tensor
#We can now check for this class in order to know if data is a multi/channel image or a label
class MultichannelImage(torch.Tensor):
    def show(self, ctx=None, **kwargs): 
        input(self.shape)
        as_numpy= np.array(self[0:3,:,:],dtype=np.uint8)
        as_numpy= as_numpy.transpose(1,2,0) #colorchannel should be last
        
        rgbim= Image.fromarray(as_numpy)
        rgbim.show()
        input(rgbim)
        

        
        
        return show_image(rgbim, title="rgbim", ctx=ctx, **kwargs)
"""


"""
class AlbumentationsTransform(DisplayedTransform):
    '''
    adapted from https://docs.fast.ai/10b_tutorial.albumentations.html

    A transfomr that only is aplied to the trainingset
    '''
    #split idx =0 means that the transform only is aplied to the trainingset
    # split idx =1 means that the transform only is aplied to the validationgset
    # order defines in what order this trasnfomr should be aplied in . If we set it to 2 and some other transfomr is set to 1 , the other transform will be aplied first
    split_idx, order = 0, 2
    def __init__(self, train_aug): store_attr() #store atributes among other thing does self.train_aug = train_aug #https://fastcore.fast.ai/basics.html

    #Transfomrms like rotation transfomrs need to be aplied o both images and labels when doing image segmentation.
    #This is assured by not defining any class for the input to the transforms encode() function
    #If we want a transformation to only be aplied to e.g the image We should define a class (e.g MultichannelImage) and use it for the input to the encode function
    #The DataBlock also need to cast its images to this class (e.g with a open_image() function that opens the path to the image and casts it to the Class)

    #Only transforms the images
    
    def encodes(self, img): #
        input("image :"+str(type(img)))

        #input("Got a MultichannelImage:"+str(img.shape))
        aug_img = self.train_aug(image=np.array(img))['image'] #aug_img = aug(image=np.array(img))['image']
        #input(aug_img)
        #input("TRANSFOMR DONE")
        return aug_img #PILImage.create(aug_img)
   
    #only Encodes the label
    def encodes(self, img:PILMask):  #
        input("label :"+str(type(img)))
        #input("Got a image or label"+str(img.shape))
        aug_img = self.train_aug(image=np.array(img))['mask']  # aug_img = aug(image=np.array(img))['image']
        # input(aug_img)
        # input("TRANSFOMR DONE")
        return aug_img  # PILImage.create(aug_img)

"""

#For some reason it looks like we need to create a separate class inhereting from itemtransform for each albumetation transfomr we want to aply
#AN alternative way of handling this might be to chain al transforms allready in the aug function we wend to the Itemtransform
#For the time beeing we use separat classes since it works. Separating the transforms this also make it easier to handle rgb-nir images in a sensible way when we want to aply transforms to images with more channels

def get_transforms(experiment_settings_dict):
    """
    param: experiment_settings_dict: dictionary holding all information about what transforms should be aplied
    para: droppable_channels e.g [4,5] for dropping out the 4th or 5th channel in an RGB-Nir-deviation-pulsewidth image
    """

    #droppable_channels=experiment_settings_dict["droppable_channels"]  # e.g [4,5] for dropping out the 4th or 5th channel in an RGB-Nir-deviation-pulsewidth image

    transforms=[]
    for transform_name in experiment_settings_dict["transforms"]:
        if "GaussNoise" == transform_name:
                transforms.append(SegmentationAlbumentationsTransformGaussNoise(split_idx=0))
        elif "Transpose" == transform_name:
                transforms.append(SegmentationAlbumentationsTransformTRanspose(split_idx=0))
        elif "RandomBrightness" == transform_name:
                transforms.append(SegmentationAlbumentationsTransformBrightness(split_idx=0))
        elif "RandomShadow" == transform_name:
                transforms.append(SegmentationAlbumentationsTransformSHADOW(split_idx=0))
        elif "ShiftScaleRotate" == transform_name:
                transforms.append(SegmentationAlbumentationsTransformSHIFTSCALEROTATE(split_idx=0))


        elif "VerticalFlip" == transform_name:
                transforms.append(SegmentationAlbumentationsVerticalFlip(split_idx=0))
        elif "HorizontalFlip" == transform_name:
                transforms.append(SegmentationAlbumentationsHorizontalFlip(split_idx=0))
        elif "RandomRotate90" == transform_name:
                transforms.append(SegmentationAlbumentationsRandomRotate90(split_idx=0))
        elif "channel_dropout" == transform_name:
                #split_idx=0 == only aply on data in trainingset
                #order=100 apply after all other transforms (especially after the -mean/std transformation in order to make sure the output is 0)
                #Normalize.from_stats has order 99 so 100 will cause the channel dropout to be aplied after the normalizations
                transforms.append(SegmentationAlbumentationsChannel_dropout(droppable_channels=experiment_settings_dict["droppable_channels"],split_idx=0, order=100))
        elif "crop" == transform_name:
                #order =0 , everything seems to work fine with orter0 but I would belive the correct nr should have been slightnly higher in order to come after rotation etc. But if I use a higher number I get errorss
                #split_idx = Nóne. we want to crop input for both traininhg and validation in order to not risk using more memory during validation an thus cause a out-of-memory crash during validation
                transforms.append(SegmentationAlbumentationsRandomCrop(size=experiment_settings_dict["crop_size"],split_idx=0,order=0)) # split_idx =0 : only apply during trainming !
                transforms.append(SegmentationAlbumentationsCentreCrop(size=experiment_settings_dict["crop_size"],split_idx=1,order=0)) # split_idx =1 : only aply during valideation ! 
        elif "channel_coruption" == transform_name:
                #split_idx=0 == only aply on data in trainingset
                #order=100 apply after all other transforms 
                #Normalize.from_stats has order 99 so 100 will cause the channel coruption to be aplied after the normalizations
                transforms.append(SegmentationAlbumentationsChannelCorruption(corruptible_channels=experiment_settings_dict["channel_coruption"], p_rotate=0.025, p_flip=0.025, split_idx=0, order=100))
        elif "windmill_conditional_corruption" == transform_name:
                #split_idx=0 == only aply on data in trainingset
                #order=100 run after the crop transform (order=0) so tiles are square for the 90 degree rotation
                #only fires when a trigger class (e.g windmill) is present in the label, then rotates DSM/DTM together in p fraction of those cases
                transforms.append(SegmentationAlbumentationsConditionalChannelCorruption(
                    corruptible_channels=experiment_settings_dict["conditional_corruption_channels"],
                    trigger_classes=experiment_settings_dict["conditional_corruption_classes"],
                    p=experiment_settings_dict.get("conditional_corruption_p", 0.5),
                    split_idx=0, order=100))
        elif "colorJitter" == transform_name:
                transforms.append(ColorJitter(split_idx=0))
        elif "downsize" == transform_name:
                #split_idx=None --> apply for both training and validation
                transforms.append(SegmentationAlbumentationsResize(split_idx=None,max_size = experiment_settings_dict["max_size"]))
        else:
            input(" no transform with name :"+str(transform_name))
    return transforms
    
def check_for_nan_in_tensor(tensor,info="info to show if there are nans"):
    """
    stops program if there are any nan in the tensor.
    shows tensor and any potential extra info sent to the function 
    used to check if the transforms introduce nan in the input
    
    :param tensor: pytorch tensor e.g a multichannel image
    :return
    """
    if torch.isnan(tensor).any():
            print("tensor:"+str(tensor))
            print("info:"+str(info))
            sys.exit("stoped excecution becaus of nan in tensor :"+str(info))    



def check_for_nan_in_numpy_array(array,info="info to show if there are nans"):
    """
    stops program if there are any nan in the tensor.
    shows tensor and any potential extra info sent to the function 
    used to check if the transforms introduce nan in the input
    
    :param array: numpy array e.g a multichannel image
    :return
    """
    if np.isnan(array).any():
            print("array:"+str(array))
            print("info:"+str(info))
            sys.exit("stoped excecution becaus of nan in tensor :"+str(info))

class SegmentationAlbumentationsChannel_dropout(ItemTransform):
    def __init__(self,droppable_channels,split_idx=0, order=100):
        ItemTransform.__init__(self,split_idx=split_idx, order=order)
        self.aug = albumentations.augmentations.dropout.channel_dropout.ChannelDropout(p=0.05,droppable_channels=droppable_channels)


    def encodes(self, x):
        img,mask = x
        
        img= np.transpose(img,(1,2,0)) #channel,y,x to ,y,x,chanel
        img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
        aug = self.aug(image=img, mask=np.array(mask))




        return ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32)), aug["mask"] #back to chanel, y,x

 

#croppping all trainingdata to a certain size in order to save memory and that way be able to enlarge batchsize. 
#randomly cropping instead of preproicessing to crop size makes it posible to handle rotations better. this way we can also produce many different crops from same trainingdata
class SegmentationAlbumentationsRandomCrop(ItemTransform):
    def __init__(self,size,split_idx, order): 
        ItemTransform.__init__(self,split_idx=split_idx, order=order)
        #p1 and allways_apply both make the transform allways happen (if split_idx has the correct value )
        self.aug =albumentations.RandomCrop(p=1,always_apply=True,height=size[0],width=size[1])
    def encodes(self, x):
        img,mask = x
        img= np.transpose(img,(1,2,0))
        img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
        aug = self.aug(image=img, mask=np.array(mask))
        return ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32)), PILMask.create(aug["mask"])
#fixed centre crop during validaiton in order to get the same input each time
class SegmentationAlbumentationsCentreCrop(ItemTransform):
    def __init__(self,size,split_idx, order):
        ItemTransform.__init__(self,split_idx=split_idx, order=order)
        #p1 and allways_apply both make the transform allways happen (if split_idx has the correct value )
        self.aug =albumentations.CenterCrop(p=1,always_apply=True,height=size[0],width=size[1])
    def encodes(self, x):
        img,mask = x
        img= np.transpose(img,(1,2,0))
        img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
        aug = self.aug(image=img, mask=np.array(mask))
        return ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32)), PILMask.create(aug["mask"])

class SegmentationAlbumentationsResize(ItemTransform):
    def __init__(self,max_size,split_idx):
        ItemTransform.__init__(self,split_idx=split_idx, order=-1)
        #p1 and allways_apply both make the transform allways happen (if split_idx has the correct value )
        self.aug =albumentations.LongestMaxSize(max_size=max_size,p=1,always_apply=True)
    def encodes(self, x):
        # Check if the input is a tuple (img, mask)
        if len(x)==2:
            img,mask = x
            img= np.transpose(img,(1,2,0))
            img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
            aug = self.aug(image=img, mask=np.array(mask))
            return ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32)), PILMask.create(aug["mask"])
        else:
            assert len(x) ==1

            img= x[0]
            img= np.transpose(img,(1,2,0))
            img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
            aug = self.aug(image=img)
            return [ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32))]

class SegmentationAlbumentationsVerticalFlip(ItemTransform):
    def __init__(self,split_idx):
        ItemTransform.__init__(self,split_idx=split_idx)
        self.aug = albumentations.VerticalFlip(p=0.05)
    def encodes(self, x):
        img,mask = x
        img= np.transpose(img,(1,2,0))
        img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
        aug = self.aug(image=img, mask=np.array(mask))
        return ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32)), PILMask.create(aug["mask"])
class SegmentationAlbumentationsRandomRotate90(ItemTransform):
    def __init__(self,split_idx):
        ItemTransform.__init__(self,split_idx=split_idx)
        self.aug = albumentations.RandomRotate90(p=0.05)
    def encodes(self, x):
        img,mask = x
        img= np.transpose(img,(1,2,0))
        img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
        aug = self.aug(image=img, mask=np.array(mask))
        return ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32)), PILMask.create(aug["mask"])
class SegmentationAlbumentationsHorizontalFlip(ItemTransform):
    def __init__(self,split_idx): self.aug = albumentations.HorizontalFlip(p=0.05)
    def encodes(self, x):
        img,mask = x
        img= np.transpose(img,(1,2,0))
        img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
        aug = self.aug(image=img, mask=np.array(mask))
        return ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32)), PILMask.create(aug["mask"])

class SegmentationAlbumentationsTransformGaussNoise(ItemTransform):
    def __init__(self,split_idx):
        ItemTransform.__init__(self,split_idx=split_idx)
        self.aug = albumentations.GaussNoise(p=0.05)
    def encodes(self, x):
    
        img,mask = x
        #check_for_nan_in_tensor(img,"before GaussNoise")
        #albumetations asume the order of the channels is h,w,channels but the tensors are channels,h,w 
        img= np.transpose(img,(1,2,0))
        img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
        aug = self.aug(image=img, mask=np.array(mask))
        
        #if the transformation introduced nan in the data, use the original data instead
        #if np.isnan(aug["image"]).any():
        #            return x
        
        
        #after transfomr is donw we need to transpose the tensor back to channels,h,w
        #check_for_nan_in_numpy_array(aug["image"],"after aug data")
        #check_for_nan_in_numpy_array(np.array(mask),"after GaussNoise target")
        return ImageBlockReplacement.MultiChannelImage.create( np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32)), PILMask.create(aug["mask"])
class SegmentationAlbumentationsTransformTRanspose(ItemTransform):
    def __init__(self,split_idx):
        ItemTransform.__init__(self,split_idx=split_idx)
        self.aug = albumentations.Transpose(p=0.05)
    def encodes(self, x):

        img,mask = x

        #check_for_nan_in_tensor(img,"before transpose")
        #albumetations asume the order of the channels is h,w,channels but the tensors are channels,h,w 
        img= np.transpose(img,(1,2,0))
        img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
        aug = self.aug(image=img, mask=np.array(mask))
        #check_for_nan_in_numpy_array(aug["image"],"after transpose")
        #if the transformation introduced nan in the data, use the original data instead
        #if np.isnan(aug["image"]).any():
        #            return x
        #after transfomr is donw we need to transpose the tensor back to channels,h,w
        #check_for_nan_in_numpy_array(aug["image"],"after aug data")
        #check_for_nan_in_numpy_array(np.array(mask),"after GaussNoise target")
        return ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(aug["image"],(2,0,1))),dtype=np.float32), PILMask.create(aug["mask"])

class SegmentationAlbumentationsTransformBrightness(ItemTransform):
    def __init__(self,split_idx):
        ItemTransform.__init__(self,split_idx=split_idx)
        self.aug = albumentations.RandomBrightness(p=0.05)
    def encodes(self, x):
        img,mask = x
        #check_for_nan_in_tensor(img,"before brigntes")
        #albumetations asume the order of the channels is h,w,channels but the tensors are channels,h,w 
        img= np.transpose(img,(1,2,0))
        img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
        aug = self.aug(image=img, mask=np.array(mask))
        #check_for_nan_in_numpy_array(aug["image"],"after brighntes")
        #if the transformation introduced nan in the data, use the original data instead
        #if np.isnan(aug["image"]).any():
        #            return x
        #after transfomr is donw we need to transpose the tensor back to channels,h,w
        #check_for_nan_in_numpy_array(aug["image"],"after aug data")
        #check_for_nan_in_numpy_array(np.array(mask),"after GaussNoise target")
        return ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32)), PILMask.create(aug["mask"])
        
        """
        if type(self.aug) is RandomShadow:
            #input("RANDSHADOW")
            
            
            #img= np.ascontiguousarray(img, dtype=np.uint8)
            #input(numpy_image.shape)
            #input(numpy_image[:,:,0:1].shape)
            img= np.transpose(img,(1,2,0))
            img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
            aug = self.aug(image=img, mask=np.array(mask))
            #return aug["image"], PILMask.create(aug["mask"])
            return np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32), PILMask.create(aug["mask"])
        if type(self.aug) is ShiftScaleRotate:
            #print("RANDOMROTATIONSHIFTZOOM")
            #input(img.shape)
            img= np.transpose(img,(1,2,0))
            aug = self.aug(image=np.array(img), mask=np.array(mask))
            #return aug["image"], PILMask.create(aug["mask"])
            return np.transpose(aug["image"],(2,0,1)), PILMask.create(aug["mask"])
         """

class ColorJitter(ItemTransform):
    """
    A class for gently changing the color : (you can experimetn with the settings here :https://explore.albumentations.ai/transform/ColorJitter (or an alternative here https://kornia.readthedocs.io/en/stable/augmentation.html) )
    TODO: also adjust the other channels (now only adjusts the 3 first channels)
    """
    def __init__(self,split_idx):
        ItemTransform.__init__(self,split_idx=split_idx)
        self.aug = albumentations.ColorJitter(brightness=[0.8,1.2], contrast=[0.8,1.2], saturation=[0.8,1.2], hue=[-0.2,0.2], p=0.2)
    def encodes(self, x):
        img,mask = x
        #check_for_nan_in_tensor(img,"before brigntes")
        #albumetations asume the order of the channels is h,w,channels but the tensors are channels,h,w
        img= np.transpose(img,(1,2,0))
        img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
        augmented_rgb = self.aug(image=img[:,:,:3], mask=np.array(mask))
        extra_channels = img[:, :, 3:]  # Fourth channel (e.g., Alpha or NIR)
        # Reattach the extra channels
        augmented_image = np.concatenate([augmented_rgb['image'], extra_channels], axis=2)
        #check_for_nan_in_numpy_array(aug["image"],"after brighntes")
        #if the transformation introduced nan in the data, use the original data instead
        #if np.isnan(aug["image"]).any():
        #            return x
        #after transfomr is donw we need to transpose the tensor back to channels,h,w
        #check_for_nan_in_numpy_array(aug["image"],"after aug data")
        #check_for_nan_in_numpy_array(np.array(mask),"after GaussNoise target")
        return ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(augmented_image,(2,0,1)),dtype=np.float32)), PILMask.create(augmented_rgb["mask"])
            
class SegmentationAlbumentationsTransformSHADOW(ItemTransform):
    """
    A class for applying shadows to multi-channel images
    Use the an updated version of the library 'albumetations' and function albumentations.RandomShadow
    The updated version is changed to handle multichannel images:





    albumentations.RandomShadow generates shadow[list of vertices] to be applied in its init() function. When we later call the aply() function, the same shadow will be aplied to RGB and GB-Nir

    albumentations.RandomShadow only handles RGB images so we first send through RGB, then GB-NIR and extract the resulting Nir channel.
    Then we combine the transformed RGB-Nir image , with the rest of the channels in the original image


    """
    def __init__(self,split_idx):
        ItemTransform.__init__(self,split_idx=split_idx)
        self.aug = albumentations.RandomShadow(p=0.05)
    def encodes(self, x):
        img,mask = x
        #albumetations asume the order of the channels is h,w,channels but the tensors are channels,h,w
        img= np.transpose(img,(1,2,0))
        img=np.array(img,dtype=np.uint8).astype(np.uint8).copy()
        aug= self.aug(image=img, mask=np.array(mask))

        """

        #ad shadow to rgb ,
        augmentedRGB = self.aug(image=img[:,:,0:3], mask=np.array(mask))
        #we base the result (iamge and mask) on the augmentation of augmentedRGB. And later ad the eugmented nir and non-augmented other channels
        aug= augmentedRGB

        #treat GB-nir as RGB
        #ad shadow to GB-Nir
        augmentedGB_NIR = self.aug(image=img[:,:,1:4], mask=np.array(mask))
        #extract the transformed NIR
        augmented_NIR = augmentedGB_NIR["image"][:,:,2:3]



        #use extra channels as they were before transformation
        output_img =img
        #gather the rgb nir and other channels in a single transformed image
        output_img[:,:,0:3] =augmentedRGB["image"] #copy transformed RGB
        output_img[:,:,3:4] =augmentedGB_NIR["image"][:,:,2:3] #only transfer the NIR channel

        aug["image"] = output_img
        
        """


        return ImageBlockReplacement.MultiChannelImage.create(np.array(np.transpose(aug["image"],(2,0,1)),dtype=np.float32)), PILMask.create(aug["mask"])
            
        
        
class SegmentationAlbumentationsChannelCorruption(ItemTransform):
    def __init__(self, corruptible_channels, p_rotate=0.1, p_flip=0.1, split_idx=0, order=100):
        """
        corruptible_channels: list of channels that can be corrupted
        p_rotate: probability of applying a random 90-degree rotation
        p_flip: probability of applying random mirroring (flip) transformations
        """
        ItemTransform.__init__(self, split_idx=split_idx, order=order)
        self.corruptible_channels = corruptible_channels

        # Define the albumentations augmentations
        self.aug = albumentations.Compose([
            albumentations.RandomRotate90(p=p_rotate),   # Random 90, 180, 270 degree rotation
            albumentations.Flip(p=p_flip),               # Random horizontal and vertical flip
        ], additional_targets={'mask': 'mask'})          # Ensure the mask is unaffected

    def encodes(self, x):
        img, mask = x
        img = np.transpose(img, (1, 2, 0))  # channel, y, x to y, x, channel
        img = np.array(img, dtype=np.uint8).astype(np.uint8).copy()

        # Apply augmentations to the specified corruptible channels
        for channel_idx in self.corruptible_channels:
            # Create a dummy image with only the single channel to corrupt
            corrupted_channel = img[..., channel_idx]

            # Apply albumentations augmentations only to the corrupted channel
            aug = self.aug(image=corrupted_channel, mask=np.array(mask))
            img[..., channel_idx] = aug["image"]  # Replace the corrupted channel in the original image

        # Convert the image back to (channels, height, width)
        img_aug = np.transpose(img, (2, 0, 1))

        # Return the corrupted image and the unmodified mask
        return ImageBlockReplacement.MultiChannelImage.create(np.array(img_aug, dtype=np.float32)), np.array(mask)

class SegmentationAlbumentationsConditionalChannelCorruption(ItemTransform):
    """
    Decorrelates height channels (DSM/DTM) from the rest of the input, but ONLY for tiles that
    contain a trigger class (e.g. windmills) in the label mask, and only in a fraction of those cases.

    Rationale: DSM/DTM are updated every ~5 years and can be outdated. Objects like windmills may be
    absent from old height data, so the model should not learn to rely on DSM/DTM to classify them.
    When the transform fires, the same random k*90 degree rotation is applied to all corruptible
    channels (so nDSM = DSM - DTM stays physically consistent, just spatially displaced), breaking the
    spatial correspondence with the RGB/mask while leaving RGB channels and the mask untouched.

    NOTE: DSM/DTM here are raw float32 heights (not 0-255), so this transform stays in float and does
    NOT cast to uint8 like the other transforms in this module.

    corruptible_channels: list of channel indices to rotate together (e.g. [8, 9] for DSM, DTM)
    trigger_classes: list of label class id(s) whose presence enables the transform (e.g. windmill code)
    p: probability of applying the rotation given a trigger class is present in the mask
    """
    def __init__(self, corruptible_channels, trigger_classes, p=0.5, split_idx=0, order=100):
        ItemTransform.__init__(self, split_idx=split_idx, order=order)
        self.corruptible_channels = corruptible_channels
        self.trigger_classes = trigger_classes
        self.p = p

    def encodes(self, x):
        img, mask = x
        mask_np = np.array(mask)

        # Only fire when a trigger class is present in the label, and then only in a fraction of cases
        trigger_present = np.isin(mask_np, self.trigger_classes).any()
        if not (trigger_present and random.random() < self.p):
            return x

        # channel, y, x -> y, x, channel ; keep FLOAT (do not clip height values to uint8)
        img = np.transpose(np.array(img, dtype=np.float32), (1, 2, 0))

        # 90 degree rotation requires square tiles; this transform must run after the crop transform
        assert img.shape[0] == img.shape[1], \
            "ConditionalChannelCorruption needs square tiles (run it after the crop transform)"

        # Same rotation for all corruptible channels so DSM/DTM stay mutually consistent
        k = random.randint(1, 3)  # 90, 180 or 270 degrees
        for channel_idx in self.corruptible_channels:
            img[..., channel_idx] = np.rot90(img[..., channel_idx], k)

        img = np.transpose(img, (2, 0, 1))
        return ImageBlockReplacement.MultiChannelImage.create(np.array(img, dtype=np.float32)), np.array(mask_np)

class SegmentationAlbumentationsTransformSHIFTSCALEROTATE(ItemTransform):
    def __init__(self,split_idx):
        ItemTransform.__init__(self,split_idx=split_idx)
        self.aug = ShiftScaleRotate(p=0.05)
    def encodes(self, x):
        if len(x)==2:
            #the transform should be done on both image and label
            img,mask = x
            #check_for_nan_in_tensor(img,"before shiftrotate")
            #albumetations asume the order of the channels is h,w,channels but the tensors are channels,h,w
            img= np.transpose(img,(1,2,0))

            aug = self.aug(image=np.array(img), mask=np.array(mask))
            #check_for_nan_in_numpy_array(aug["image"],"after shiftrotate")
            #if the transformation introduced nan in the data, use the original data instead
            #if np.isnan(aug["image"]).any():
            #            return x
            #after transfomr is donw we need to transpose the tensor back to channels,h,w
            #check_for_nan_in_numpy_array(aug["image"],"after aug data")
            #check_for_nan_in_numpy_array(np.array(mask),"after GaussNoise target")
            return ImageBlockReplacement.MultiChannelImage.create(np.transpose(aug["image"],(2,0,1))), PILMask.create(aug["mask"])
        else:
            #the transform should only be aplied to an image
            img= x[0]
            #check_for_nan_in_tensor(img,"before shiftrotate")
            #albumetations asume the order of the channels is h,w,channels but the tensors are channels,h,w
            as_np = np.array(img)
            #debug1=as_np.shape

            img= as_np #np.transpose(as_np,(1,2,0))

            debug2=img.shape
            aug = self.aug(image=np.array(img))
            #check_for_nan_in_numpy_array(aug["image"],"after shiftrotate")
            #if the transformation introduced nan in the data, use the original data instead
            #if np.isnan(aug["image"]).any():
            #            return x
            #after transfomr is donw we need to transpose the tensor back to channels,h,w
            #check_for_nan_in_numpy_array(aug["image"],"after aug data")

            return ImageBlockReplacement.MultiChannelImage.create(aug["image"],(2,0,1))
    
def set_seed(dls,x=42): #must have dls, as it has an internal random.Random
    """
    In order to sontroll the randomness ascosiated with batch order and transforms
    https://github.com/fastai/fastai/issues/2832#issuecomment-698716520
    """
    #we want the same image to be selected everytime
    inactivate_dls_randomness= True
    if inactivate_dls_randomness:
        dls.rng.seed(x) #added this line
    #we dont want the same transform to be aplied
    inactivate_more_randomness= False
    if inactivate_more_randomness:
        random.seed(x)
        np.random.seed(x)
        torch.manual_seed(x)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)        

def visualize_transforms(experiment_settings_dict,image,save_images):


    if image:
        (means,stds) =  (experiment_settings_dict["means"],experiment_settings_dict["stds"]) #imagenet_stats
        #we only want to visualize RGB
        as_np_raw=np.array(Image.open(image))[:,:,0:3]
        means = means[0:3]
        stds = stds[0:3]
        as_np_divided = as_np_raw/255
        print("after / 255 "+str(as_np_divided))
        as_np_meaned = as_np_divided -means
        print("after as_np -means "+str(as_np_meaned))
        as_np_stded = as_np_meaned /stds
        print("after / stds "+str(as_np_stded))
        #if an image is provided we visualize how the image will look like after the transforms are provided
        #the simplest way to look a a certain image after it has been sent through the pipeline is to create a trainer and send aan image through classification
        infer.ad_values_nececeary_for_dataset_loader_creation(experiment_settings_dict)
        dls = sdfi_dataset.get_dataset(experiment_settings_dict)
        #set_seed(dls) when visualizing a single image we want to aply different transforms every time
        training= train.basic_traininFastai2(experiment_settings_dict,dls)
        #load saved weights
        training.learn.load(str(experiment_settings_dict["model_to_load"]).rstrip(".pth"))
        print("####################################################################################")
        print("######################NOW PREDICTING ON THE IMAGE###########################")
        print("image:"+str(image))
        
        print("as_np_raw "+str(as_np_raw))
        print("as_np_divided "+str(as_np_divided))
        print("as_np_meaned "+str(as_np_meaned))
        print("as_np_meaned "+str(as_np_meaned))
        dl = training.learn.dls.test_dl([pathlib.Path(image)]) # dl = training.learn.dls.test_dl(all_files) # 
        #Does not work with 4 chanel tensors
        #if show:
        #    dl.show_batch()
        
        #sending with_input=True to get_preds() makes it return the input together with the predictions 
        
        
        preds = training.learn.get_preds(dl=dl,with_input=true)
        
        the_input= preds[0]
        the_prediction =  preds[1][0]
        
        
        #show input
        

        
        #change from index_in_batch,channels ,height,width to height,width,channels
        the_input = the_input.squeeze() #remove batch dimension
        the_input=the_input.permute([1,2,0]) ##change from channels ,height,width to height,width,channels
       
        rgb= the_input[:,:,0:3]
        
        print("after removal of 4th channel"+str(rgb))
        
        #reversing the transforms in order to look at the original image
        im_np = np.array(((rgb*np.array(stds))+np.array(means))*255)
        print("im_np before show "+ str(im_np))
        im_np= np.clip(im_np, 0, 255)
        im_np=np.array(im_np,dtype=np.uint8)

        #print("im_np.max()"+str(im_np.max()))
        #print("im_np.min()"+str(im_np.min()))
        print("showing original image without nir channel")
        Image.fromarray(as_np_raw).show()
        print("showing image after it has beeen sent through the data-pipeline and back to reconstructed data-range ")
        Image.fromarray(im_np).show()
        print("showing difference between original image and data augmentated image")
        Image.fromarray(im_np-as_np_raw).show()




    else:
        dls = sdfi_dataset.get_dataset(experiment_settings_dict)
        set_seed(dls)
        #if no image has been provided we simply show some images from the dataset
        
        #get the image from the first image-label pair in the  dataset (after transforms have been aplied)
        for i in range(10):
            (means,stds) =  (experiment_settings_dict["means"],experiment_settings_dict["stds"]) #imagenet_stats
            print("loading the dataset..")
            
            
            #print(next(iter(dls.train)))
            a_batch = next(iter(dls.train)) # dls.train.one_batch()
            
            a_batch_of_images =  a_batch[0]
            a_batch_of_labels =  a_batch[1]
            for j in range(1):
            
                first_image= np.array(a_batch_of_images[0].cpu())
                first_label= np.array(a_batch_of_labels[0].cpu())

                first_image=first_image.transpose([1,2,0])

                save_rgbnir_lidar_image_to_disk = True
                if save_rgbnir_lidar_image_to_disk:
                    #INSPECTING RGB-Nir-lidar
                    try:
                        rgbnirlidar_np = np.array(((first_image*np.array(stds))+np.array(means))*255)
                        rgbnirlidar_np= np.clip(rgbnirlidar_np, 0, 255)
                        rgbnirlidar_np=np.array(rgbnirlidar_np,dtype=np.uint8)

                        lidar = first_image[:,:,4]
                        lidar = lidar -lidar.min()
                        lidar = lidar/lidar.max()
                        lidar = lidar *255
                        Image.fromarray(lidar).save("normalizedlidar.tif")

                        Image.fromarray(np.array(rgbnirlidar_np,dtype=np.uint8)[:,:,0:4]).save("transformed_image_RGBnir.tif")
                        Image.fromarray(np.array(rgbnirlidar_np,dtype=np.uint8)[:,:,0:3]).save("transformed_image_RGB.tif")
                        Image.fromarray(np.array(rgbnirlidar_np,dtype=np.uint8)[:,:,4]).save("transformed_image_lidar.tif")
                    except:
                        print("failed to visualize extra channels")


                #INSPECTING RGB
                rgb= first_image[:,:,0:3]
                means = means[0:3]
                stds = stds[0:3]

                im_np = np.array(((rgb*np.array(stds))+np.array(means))*255)

                im_np= np.clip(im_np, 0, 255)
                im_np=np.array(im_np,dtype=np.uint8)
                if save_images:
                    Image.fromarray(im_np).save("image_"+str(i)+str(j)+".jpg")
                else:
                    Image.fromarray(im_np).show()






                label_data = np.array(first_label,dtype=np.uint8)
                #input(image_data.max())
                #input(image_data.min())
            



                show_label_image = True

                wait_for_enter = False
                if show_label_image:
                    if save_images:
                        # Define the colormap and norm
                        cmap = cm.get_cmap('tab20', 13)  # 'tab20' colormap with 13 unique colors
                        norm = mcolors.Normalize(vmin=0, vmax=12)

                        # Apply the colormap to the label data
                        colored_image = cmap(norm(label_data))

                        # Remove the alpha channel and convert to 8-bit (0-255) RGB format
                        colored_image_rgb = (colored_image[..., :3] * 255).astype(np.uint8)


                        Image.fromarray(colored_image_rgb).save("label_"+str(i)+str(j)+".jpg")
                        #plt.imshow(label_data,cmap="tab20",vmin=0, vmax=10)
                        #plt.axis('off')  # Turn off the axis
                        # Saving the plot to disk without any additional elemnts to make it easier to compare to the input data
                        #plt.savefig("label_"+str(i)+".jpg",bbox_inches='tight', pad_inches=0, format='jpg')
                    else:
                        plt.imshow(label_data,cmap="tab20",vmin=0, vmax=10)
                        plt.show()


                if wait_for_enter:
                    input("PRES ENTER FOR NEXT IMAGE")
            
        #for i in range(10):
        #    dls.show_batch(rows=2, figsize=(7,5))



if __name__ == "__main__":
    """
    Given one or more config-files that defines a training or prediction,
    visualize how images will be transformed during training 
    
    Ways of calling it below:
    
    #verifying that the /255 mean and std normalisations aplied in a good way by predicting on an image (OBS I have not been able to aply the transforms in a good way during inference) 
    #meant to be used like this 
    #python sdfi_transforms.py --config T:\config_files\transforms_befestelse_orthoimages_iter_7.ini --image "T:\trainingdata\befastelse\orthoimages_iteration_3\images\2022_1km_6215_455_0_0.png"
    
    #if you want to check the transformations you should use a .ini file for a training like this 
    #python sdfi_transforms.py --config T:\config_files\train_befestelse_orthoimages_iter_7.ini
        
    """
    
    usage_example="example usage: \n "+r"python sdfi_transforms.py --config T:\config_files\train_befestelse_orthoimages_iter_9.ini"
    # Initialize parser
    parser = argparse.ArgumentParser(
        epilog=usage_example,
        formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument("-c", "--config", help="one or more paths to experiment config file",nargs ='+',required=True)
    parser.add_argument("-i", "--image", help="one image to send through datapipeline (inclusive transforms)",required=False)
    parser.add_argument('--save_images', action='store_true', default=False)

    args = parser.parse_args()


    for config_file_path in args.config:
        experiment_settings_dict= sdfi_utils.load_settings_from_config_file(config_file_path)
        visualize_transforms(experiment_settings_dict,image=args.image,save_images=args.save_images)

