#!/usr/bin/env python
# coding: utf-8
"""
Train a model to do semantic segmentation
"""
import sdfi_dataset
import utils.utils as sdfi_utils
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
from fastai.layers import  _get_norm
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
import shutil
import utils.utils as sdfi_utils
from wwf.vision.timm import *
from utils.timm_unet_utils import timm_unet_splitter
from fastai.vision.all import GradientAccumulation

from fastai.callback.core import Callback

from fastai.learner import Callback, CancelBatchException, CancelEpochException


def make_deterministic():
    print("making the training repeatable so that differetn runs easier can be compared to each other")
    print("as part of this , num workersis set to 1")
    # Set environment variables for deterministic behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = '0'

    # Set seeds for Python, NumPy, and PyTorch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Set PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class CSVLoggerWithLR(CSVLogger):
    def after_epoch(self):
        # Get the current learning rates from the optimizer
        current_lrs = [group['lr'] for group in self.learn.opt.param_groups]

        # Extend the log with learning rates
        log_values = self.learn.recorder.log + current_lrs

        # Ensure the header includes learning rate columns
        if not hasattr(self, 'header_written') or not self.header_written:
            lr_headers = [f'lr_{i}' for i in range(len(current_lrs))]
            self.file.write(','.join(self.learn.recorder.metric_names + lr_headers) + '\n')
            self.header_written = True

        # Write the log values to the CSV file
        self.file.write(','.join(map(str, log_values)) + '\n')
        self.file.flush()

class DoThingsAfterBatch(Callback):
    """
    Save model after n batch
    """

    def __init__(self, n_batch: int = 200_000):
        """
        :param n_batch: Save model after n batch in a epoc
        """
        self.iter_string = "batch_string_NOT_set"
        self._modulus_faktor = n_batch


    def after_batch(self):
        if self._modulus_faktor < 2:
            return
        if self.iter % self._modulus_faktor == (self._modulus_faktor - 1):
            print("Iter: {}of{}".format(self.iter, self.n_iter))
            # save_name = self.experiment_settings_dict["job_name"]
            self.iter_string = "Batch_model_{}_{}".format(self.epoch, self.iter)
            x_cpu = self.loss.cpu()
            self.lr_string = "  loss={}".format(x_cpu.detach().numpy())
            print("Batch save filename:" + self.iter_string + self.lr_string)
            self.learn.save(self.iter_string)
            print("Batch model gemt!")


# SegFormer model wrapper for fastai
class SegFormerWrapper(nn.Module):
    """
    Wrapper for SegFormer models from transformers library to work with fastai
    """
    def __init__(self, model_name, num_classes, n_in=3, pretrained=True, ignore_index=255):
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        if pretrained:
            # Load pretrained model
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            
            # If n_in != 3, we need to adapt the first conv layer
            if n_in != 3:
                self._adapt_input_channels(n_in)
        else:
            # Create from config
            config = SegformerConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)
            
            if n_in != 3:
                self._adapt_input_channels(n_in)
    
    def _adapt_input_channels(self, n_in):
        """Adapt the first convolution to accept n_in channels instead of 3"""
        # SegFormer's first layer is in the patch embedding
        old_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        
        # Create new conv with n_in channels
        new_conv = nn.Conv2d(
            n_in, 
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Initialize new conv
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        if new_conv.bias is not None:
            nn.init.constant_(new_conv.bias, 0)
        
        # If pretrained and expanding channels, copy weights for first 3 channels
        if n_in >= 3 and old_conv.weight.shape[1] == 3:
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
        
        # Replace the conv layer
        self.model.segformer.encoder.patch_embeddings[0].proj = new_conv
    
    def forward(self, x):
        # SegFormer expects input in NCHW format and returns logits
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        
        # Upsample logits to match input size if needed
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        return logits


def create_segformer_learner(dls, model_name, loss_func, metrics, wd=1e-2, 
                             path='.', model_dir='models', n_in=3, 
                             num_classes=None, pretrained=True, ignore_index=255):
    """
    Create a fastai learner with SegFormer model
    
    Args:
        dls: fastai DataLoaders
        model_name: SegFormer model name (e.g., 'nvidia/segformer-b0-finetuned-ade-512-512')
        loss_func: Loss function
        metrics: Metrics to track
        wd: Weight decay
        path: Path for logs
        model_dir: Directory for saving models
        n_in: Number of input channels
        num_classes: Number of output classes (inferred from dls if None)
        pretrained: Whether to use pretrained weights
        ignore_index: Index to ignore in loss calculation
    """
    if num_classes is None:
        # Try different ways to get number of classes from DataLoaders
        if hasattr(dls, 'c'):
            num_classes = dls.c
        elif hasattr(dls, 'vocab'):
            num_classes = len(dls.vocab)
        elif hasattr(dls.train, 'c'):
            num_classes = dls.train.c
        else:
            raise ValueError("Could not infer number of classes from DataLoaders. Please provide num_classes explicitly.")
    
    # Create the SegFormer model
    model = SegFormerWrapper(
        model_name=model_name,
        num_classes=num_classes,
        n_in=n_in,
        pretrained=pretrained,
        ignore_index=ignore_index
    )
    
    # Create the learner
    learn = Learner(
        dls=dls,
        model=model,
        loss_func=loss_func,
        metrics=metrics,
        wd=wd,
        path=path,
        model_dir=model_dir
    )
    
    return learn


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

        return lr_valley

    
    def create_folders(self):
        """
        create folders needed to save models and logs
        """
        #make sure the path/to/folder that the model should be saved in exists

        pathlib.Path(self.experiment_settings_dict["model_folder"]).mkdir(parents=True, exist_ok=True)
        #make sure the path/to/folder that the model should be saved in exists
        pathlib.Path(self.experiment_settings_dict["log_folder"]).mkdir(parents=True, exist_ok=True)




    def train(self,lr_max):
        """
        :param lr_max: the max learningrate that the fit_one_cyckle will cykle towards and back from.
        #if fixedlearning rate is used, lr_max will be used for the complete training
        :return: None

        Trains a model on a dataset
        """
        self.create_folders()

        
        
        
        if self.experiment_settings_dict["model_to_load"]:
            print("loading :"+str(self.experiment_settings_dict["model_to_load"]))    
            #load save weights
            self.learn.load(str(Path(self.experiment_settings_dict["model_to_load"]).with_suffix("")))
        else:
            print("no model to load..")    

        
        if "save_on_batch_iter_modulus_n" in self.experiment_settings_dict:
            n_batch = self.experiment_settings_dict["save_on_batch_iter_modulus_n"]
        else:
            n_batch = 0

        if self.experiment_settings_dict["freeze"]:
            self.learn.freeze()
        else:
            self.learn.unfreeze()
        #IN order to monitor the acuracy we might have to modifie the learner as they talk about here https://forums.fast.ai/t/equivalent-of-add-metrics-in-fastai2/77575/2
        #In order to skip to a a certain epoch before training (and get the Lr of the apropriate part of the LR-scedule we ad a skip_to_epoch callback)
        #If you get problems with nan in training loss it might be a good idea to ad 'GradientClip(0.1)' to the csbs list
        if self.experiment_settings_dict["last_epoch"]>0:
            start_epoch=self.experiment_settings_dict["last_epoch"]+1
        else:
            start_epoch =0


        if self.experiment_settings_dict["sceduler"] =="fit_one_cycle":
            self.learn.fit_one_cycle(n_epoch=self.experiment_settings_dict["epochs"],start_epoch=start_epoch, lr_max=lr_max,
                                     cbs=[GradientAccumulation(self.experiment_settings_dict["n_acc"]),GradientClip(self.experiment_settings_dict["gradient_clip"]),SaveModelCallback(with_opt=True,every_epoch= True, monitor='valid_loss', fname=self.experiment_settings_dict["job_name"]),
                                          CSVLoggerWithLR(fname= self.experiment_settings_dict["job_name"]+".csv", append=True),
                                          DoThingsAfterBatch(n_batch=n_batch)
                                          ])
        elif self.experiment_settings_dict["sceduler"] =="fixed":
            self.learn.fit(n_epoch=self.experiment_settings_dict["epochs"],start_epoch=start_epoch, lr=lr_max,
                           cbs=[SaveModelCallback(with_opt=True,every_epoch= True, monitor='valid_loss', fname=self.experiment_settings_dict["job_name"]),
                                CSVLogger(fname= self.experiment_settings_dict["job_name"]+".csv", append=True),
                                DoThingsAfterBatch(n_batch=n_batch)
                                ])
        else:
            sys.exit("no valid learning rate sceduler!")



        print("saving model")
        self.learn.save(self.experiment_settings_dict["job_name"])




    def get_basic_training(self,experiment_settings_dict,dls):
        """
        :param experiment_settings_dict: a dictionary holding the parameters for the training
        :param dls: a dataloader feed to the unet learner
        :return: a unet learner that use data from the dataloader
        """
        if "ignore_index" in experiment_settings_dict:
            ignore_index = int(experiment_settings_dict["ignore_index"])
        else:
            ignore_index=255
        print("USING ignore index :"+str(ignore_index))

        
        
        def valid_accuracy(inp, targ):
            """
            valid_accuracy runs after epoch, calculating the accuracy on the validationset
            Dokumentation that shows that the metrics per default is computed only on the validationset can be ound here: https://docs.fast.ai/learner.html#Learner
            """
            targ = targ.squeeze(1)
            
            void_code=ignore_index #use a code that does not exist in the dataset
            #print("void sum:"+str((targ == void_code).sum()))
            #the masked target is the same as the target (we dont use 'dont care labels')
            mask = targ != void_code
            #print("input:"+str(inp))
            #print("targ:"+str(targ[])) 
            return (inp.argmax(dim=1)[mask] == targ[mask]).float().mean()

        
            
        #the default loss function for a fastai2 unet is a flattened CrossEntropyloss. We here define it ourself so we can modify the ignore_index (what label should be ignored when computing the loss)
        #https://forums.fast.ai/t/loss-function-of-unet-learner-flattenedloss-of-crossentropyloss/51605
        #https://docs.fast.ai/losses.html#CrossEntropyLossFlat

        if "class_weights" in experiment_settings_dict and experiment_settings_dict["class_weights"]:
            weights = torch.tensor(experiment_settings_dict["class_weights"]).cuda()
            print("Using class weights: {}".format(experiment_settings_dict["class_weights"]))
            a_loss_func= CrossEntropyLossFlat(axis=1,ignore_index=ignore_index,weight=weights) 
        else:
            print("weighting all classes equally!")
            a_loss_func= CrossEntropyLossFlat(axis=1,ignore_index=ignore_index)

        # Check if using SegFormer
        value = self.experiment_settings_dict["model"]
        is_segformer = isinstance(value, str) and value.startswith("segformer")
        
        if is_segformer:
            print("Building SegFormer learner...")
            
            # Map common SegFormer model names to HuggingFace model IDs
            segformer_models = {
                "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
                "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
                "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
                "segformer-b3": "nvidia/segformer-b3-finetuned-ade-512-512",
                "segformer-b4": "nvidia/segformer-b4-finetuned-ade-512-512",
                "segformer-b5": "nvidia/segformer-b5-finetuned-ade-640-640",
            }
            
            model_name = segformer_models.get(
                self.experiment_settings_dict["model"], 
                self.experiment_settings_dict["model"]
            )
            
            pretrained = experiment_settings_dict.get("pretrained", True)
            
            # Try to get number of classes from experiment settings or codes.txt file
            num_classes = None
            if "num_classes" in experiment_settings_dict:
                num_classes = experiment_settings_dict["num_classes"]
            elif "n_classes" in experiment_settings_dict:
                num_classes = experiment_settings_dict["n_classes"]
            elif "path_to_codes" in experiment_settings_dict:
                # Load number of classes from codes.txt file
                codes_path = experiment_settings_dict["path_to_codes"]
                try:
                    with open(codes_path, 'r') as f:
                        class_names = [line.strip() for line in f.readlines() if line.strip()]
                    num_classes = len(class_names)
                    print(f"Loaded {num_classes} classes from {codes_path}: {class_names}")
                except Exception as e:
                    print(f"Warning: Could not read codes file from {codes_path}: {e}")
            
            learn = create_segformer_learner(
                dls=dls,
                model_name=model_name,
                loss_func=a_loss_func,
                metrics=valid_accuracy,
                wd=1e-2,
                path=self.experiment_settings_dict["log_folder"],
                model_dir=self.experiment_settings_dict["model_folder"],
                n_in=len(experiment_settings_dict["means"]),
                num_classes=num_classes,
                pretrained=pretrained,
                ignore_index=ignore_index
            )
            
        elif "bottleneck" in self.experiment_settings_dict or  self.experiment_settings_dict["model"] in ["efficientnetv1_m","efficientnetv2_m","efficientnetv2_l","efficientnetv2_rw_s.ra2_in1k","efficientnetv2_rw_m.agc_in1k","tf_efficientnetv2_l.in21k","tf_efficientnetv2_xl.in21k"]:
            #using a timm_learner from the wwf library (walk faster with fastai)
            print("building tim based unet learner with wwtf library...")

            pretrained=True


            learn = timm_unet_learner(dls, self.experiment_settings_dict["model"], loss_func=a_loss_func,metrics=valid_accuracy,bottleneck=experiment_settings_dict["bottleneck"], wd=1e-2,
                             path= self.experiment_settings_dict["log_folder"],pretrained=pretrained,
                             model_dir=self.experiment_settings_dict["model_folder"] ,n_in=len(experiment_settings_dict["means"]),
                             splitter=timm_unet_splitter)
        else:
            # fastai asumes 'model_dir' to be a path that is relative to 'path'. In order to make model_dir independent of 'path' we need to make the model_dir path absolute with 'resolve()' first.
            learn = unet_learner(dls, self.experiment_settings_dict["model"], loss_func=a_loss_func,metrics=valid_accuracy, wd=1e-2,
                             path= self.experiment_settings_dict["log_folder"],
                             model_dir=self.experiment_settings_dict["model_folder"] ,n_in=len(experiment_settings_dict["means"]))#callback_fns=[partial(CSVLogger, filename= experiment_settings_dict["job_name"], append=True)])
        
        if self.experiment_settings_dict["to_fp16"]:
            print("training with mixed precision")
            return learn.to_fp16()
        else:
            print("not training with mixed precision!!")
            return learn




def train(experiment_settings_dict):
    """
    :param experiment_settings_dict: a dictionary holding the parameters for the training
    :return: None

    1.loads a dataset
    2.configures a unet-training
    3.sets a learningrate 
    4.ev loads pretrained model-weights
    4.trains
    """
    run_on_cpu = False
    if run_on_cpu:
        defaults.device = torch.device('cpu')

    print("loading the dataset..")
    dls = sdfi_dataset.get_dataset(experiment_settings_dict)

    # Define a transfomr that just prints out the 
    '''
    class PrintStatsTransform(Transform):
        order = 200  # Ensures this runs last
        max =0
        min = 100
        def encodes(self, x: torch.Tensor):
            # Apply only if the input is a Tensor with 4 dimensions (batch of images)
            if isinstance(x, torch.Tensor) and x.ndim == 4:
                # Calculate statistics
                print(np.array(x.cpu()))
                print(x.shape)
                for i in range(11):
                    print(f"min Channel {i}:")
                    print(np.array(x.cpu())[:,i,:,:].flatten().min())
                    if np.array(x.cpu())[:,i,:,:].flatten().min() < self.min:
                        self.min = np.array(x.cpu())[:,i,:,:].flatten().min()


                for i in range(11):
                    print(f"max Channel {i}:")
                    print(np.array(x.cpu())[:,i,:,:].flatten().max())
                    if np.array(x.cpu())[:,i,:,:].flatten().max() > self.max:
                        self.max = np.array(x.cpu())[:,i,:,:].flatten().max()


                for i in range(11):
                    print(f"mean Channel {i}:")
                    print(np.array(x.cpu())[:,i,:,:].flatten().mean())

                for i in range(11):
                    print(f"std Channel {i}:")
                    print(np.array(x.cpu())[:,i,:,:].flatten().std())

                print("min encountered = "+str(self.min))
                print("max encountered = "+str(self.max))

                # Print the statistics
                #print(f"Max: {max_val:.4f}, Min: {min_val:.4f}, Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                # Return the input unchanged
            return x
    # Add the custom transform to the `after_batch` pipeline
    dls.after_batch.add(PrintStatsTransform())
    '''
 
    print("setting up a unet training..")
    training= basic_traininFastai2(experiment_settings_dict,dls)

    if "lr" in experiment_settings_dict:
        max_lr = experiment_settings_dict["lr"]
        print("using predefined max learning rate :"+str(max_lr))
        
    else:
        print("Finding learning rate...")
        lr_valley= training.find_learning_rate(show_images=False)

        if experiment_settings_dict["sceduler"] == "fit_one_cycle":
            multiply_with=30
            print("multiplying lr_valley with " + str(multiply_with) + "to get a max_lr value compatible with fit_one_cycle")
            max_lr = lr_valley*multiply_with

        elif experiment_settings_dict["sceduler"] == "fixed":
            print("using lr_valley as it is as learning rate when training with fixed learning rate (scedule = 'fixed' == no lr scedule)")
            max_lr = lr_valley
        else:
            sys.exit("'sceduler' should be 'fit_one_cycle' or 'fixed'")
        #store the found lr in the dictionary so we can save it to .json file in the log folder
        experiment_settings_dict["lr_finder_lr"] = max_lr

    print("max_lr:"+str(max_lr))


    print("run the training..")
    print("job_name: " + experiment_settings_dict["job_name"])

    training.train(lr_max=max_lr)

    print("TRAINING DONE! job_name: " + experiment_settings_dict["job_name"])
    sdfi_utils.save_dictionary_to_disk(experiment_settings_dict)






def infer_model_and_log_folders(experiment_settings_dict):
    """
    :param experiment_settings_dict: a dictionary holding the parameters for the training
    :return: None

    creates 'model_folder' and 'log_folder' entries in the dictionary based on the value for 'experiment_root' and 'job_name'
    """
    # fastai asumes 'model_folder' to be a path that is relative to 'log_folder'. In order to make it relative to the location the script is run from we need to make it absolute with 'resolve()' first.
    experiment_settings_dict['model_folder']=( Path(experiment_settings_dict['experiment_root'])/Path(experiment_settings_dict['job_name'])/Path("models") ).resolve()
    experiment_settings_dict['log_folder']=(Path(experiment_settings_dict['experiment_root'])/Path(experiment_settings_dict['job_name'])/Path("logs")).resolve()
   



if __name__ == "__main__":
    """
    Given one or more config-files that defines a training,
    Trains a model on a dataset.
    
    
    """
    usage_example="example usage: \n "+r"python train.py --config configs/example_configs/train_example_dataset.ini \n to use anotehr GPU than 0 call it like this \n CUDA_VISIBLE_DEVICES= python train.py   "
    # Initialize parser
    parser = argparse.ArgumentParser(
                                    epilog=usage_example,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument("-c", "--config", help="one or more paths to experiment config file",nargs ='+',required=True)
    parser.add_argument('--deterministic', action='store_true', help='tries to make different trainings comparable and repeatable')

    args = parser.parse_args()


    for config_file_path in args.config:
        experiment_settings_dict= sdfi_utils.load_settings_from_config_file(config_file_path)
        if args.deterministic:
            experiment_settings_dict["num_workers"]= 1
            make_deterministic()

        infer_model_and_log_folders(experiment_settings_dict)
        train(experiment_settings_dict=experiment_settings_dict)
