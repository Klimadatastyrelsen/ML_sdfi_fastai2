import ML_sdfi_fastai2.sdfi_dataset as sdfi_dataset
import ML_sdfi_fastai2.train as train
import ML_sdfi_fastai2.utils.utils as sdfi_utils
import argparse
import pathlib
from fastcore.xtras import Path
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import rasterio
import time
import torch
import sys
from torch.multiprocessing import Process, Queue
from fastai.vision.all import *

#OBS! for some reasone spawn is nececeary during inference but not alowed during training!
#torch.multiprocessing.set_start_method('spawn') # otherwise we get : Error	"Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method"
#if not torch.multiprocessing.get_start_method(allow_none=True):
#    #this is nly alowed to be set once. and the if statement above makes sure its only set once
#    torch.multiprocessing.set_start_method('spawn') # otherwise we get : Error      "Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spa>


# Custom callback to save segmentation output
class SaveSegmentationOutput(Callback):
    def __init__(self, learn, save_path,batch_filenames,data_to_save_queue,experiment_settings_dict):
        super().__init__()
        self.learn = learn
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.batch_filenames = batch_filenames
        self.data_to_save_queue = data_to_save_queue
        self.experiment_settings_dict = experiment_settings_dict
        self.start_time = time.time()


    def after_pred(self,verbose=False):
        #print("room left in queue: "+str(self.data_to_save_queue.maxsize - self.data_to_save_queue.qsize()))
        call_back_start = time.time()
        #print("CALLBACK IS RUNNING!")
        if verbose:print("self.learn.n_iter:"+str(self.learn.n_iter))
        if verbose:print("self.learn.iter:"+str(self.learn.iter))
        if verbose:print("Time per image for inference: "+str((time.time()-self.start_time )/((self.learn.iter+1)*self.experiment_settings_dict["batch_size"])))






        #now tryong to save probabilities in teh same way"
        #after aplying softmax I get identicall numbers for the first 10 numbers! (obs no log)
        batch_probs = torch.nn.functional.softmax(self.learn.pred).cpu().numpy()



        filenames= self.batch_filenames[self.learn.iter]
        for i in range(len(batch_probs)):
            fname = filenames[i]
            input_data_path = fname
            # Use batch index to get the corresponding prediction
            probs = batch_probs[i]
            file_stem = Path(Path(str(fname)).name).stem
            #print("calback saving to : "+str(self.save_path / f"{file_stem}_callback_prob.png"))
            file_name = fname.name
            self.data_to_save_queue.put((probs,input_data_path ,self.save_path/file_name,self.experiment_settings_dict))



            '''
            with rasterio.open(fname) as src:
                # make a copy of the geotiff metadata so we can save the prediction/probabilities as the same kind of geotif as the input image
                new_meta = src.meta.copy()
                new_xform = src.transform

            if self.experiment_settings_dict["crop_size"]:
               y_index_start =int((probs.shape[1]- self.experiment_settings_dict["crop_size"])/2)
               x_index_start = int((probs.shape[2]- self.experiment_settings_dict["crop_size"])/2)
               # create a translation transform to shift the pixel coordinates
               crop_translation = rasterio.Affine.translation(x_index_start, y_index_start)
               # prepend the pixel translation to the original geotiff transform
               new_xform = new_xform * crop_translation
               new_meta['width'] = int(self.experiment_settings_dict["crop_size"])
               new_meta['height'] = int(self.experiment_settings_dict["crop_size"])
               new_meta['transform'] = new_xform
               #set the number of channels in the output
               new_meta["count"]=probs.shape[0]
               y_index_end = y_index_start+int(self.experiment_settings_dict["crop_size"])
               x_index_end = x_index_start+int(self.experiment_settings_dict["crop_size"])
               probs= probs[:,y_index_start:y_index_end,x_index_start:x_index_end]
            file_name = fname.name
            self.data_to_save_queue.put((probs, self.save_path/file_name,new_meta,self.experiment_settings_dict))
            '''

            #save_probabilities_as_uint8_no_queue(probs=probs, path_to_probabilities= str(self.save_path / f"{file_stem}_callback_prob.png"),new_meta=new_meta)
        if verbose:print("callback took: "+str(time.time() - call_back_start))

def save_probabilities_as_float32(probs,path_to_probabilities,new_meta):
    """
    If we dont care about memory usage or hard disc usage we can store probs in float32
    """



    # probabilities are floats
    new_meta["count"] = probs.shape[0]
    new_meta["dtype"] = np.float32

    with rasterio.open(path_to_probabilities, "w", **new_meta) as dest:
        dest.write(np.array((probs),dtype=np.float32))

def save_probabilities_as_uint8_no_queue(probs,path_to_probabilities,new_meta):
    """
    not using a separate process for saving data
    """



    # probabilities are floats
    new_meta["count"] = probs.shape[0]
    new_meta["dtype"] = np.uint8

    with rasterio.open(path_to_probabilities, "w", **new_meta) as dest:
        dest.write(np.array((probs*255),dtype=np.uint8))

def save_probabilities_as_uint8(queue,verbose= False):
    #(probs,path_to_probabilities,new_meta):
    """
    Probabilities can be saved in uint8 format in order to save space and memory usage
    """
    totall_time_spent_saving =0
    try:
        while True:
            data_from_queue = queue.get()
            #print("recived data:"+str(data_from_queue))
            if data_from_queue is None:
                if verbose:print("totall time spent saving:"+str(totall_time_spent_saving))
                break  # End the loop when None is received
            saving_data_start_time = time.time()
            (probs,input_data_path ,path_to_probabilities,experiment_settings_dict) = data_from_queue

            with rasterio.open(input_data_path) as src:
                # make a copy of the geotiff metadata so we can save the prediction/probabilities as the same kind of geotif as the input image
                new_meta = src.meta.copy()
                new_xform = src.transform

            if experiment_settings_dict["crop_size"]:
               y_index_start =int((probs.shape[1]- experiment_settings_dict["crop_size"])/2)
               x_index_start = int((probs.shape[2]- experiment_settings_dict["crop_size"])/2)
               # create a translation transform to shift the pixel coordinates
               crop_translation = rasterio.Affine.translation(x_index_start, y_index_start)
               # prepend the pixel translation to the original geotiff transform
               new_xform = new_xform * crop_translation
               new_meta['width'] = int(experiment_settings_dict["crop_size"])
               new_meta['height'] = int(experiment_settings_dict["crop_size"])
               new_meta['transform'] = new_xform
               #set the number of channels in the output
               new_meta["count"]=probs.shape[0]
               y_index_end = y_index_start+int(experiment_settings_dict["crop_size"])
               x_index_end = x_index_start+int(experiment_settings_dict["crop_size"])
               probs= probs[:,y_index_start:y_index_end,x_index_start:x_index_end]



            #(probs,path_to_probabilities,new_meta,experiment_settings_dict) = data_from_queue
            '''

            # probabilities are floats scaled by multiplication and converted to uint8
            new_meta["count"] = probs.shape[0]
            new_meta["dtype"] = np.uint8

            with rasterio.open(path_to_probabilities, "w", **new_meta) as dest:
                dest.write(np.array((probs *255),dtype=np.uint8))
            took=time.time()-saving_data_start_time
            print("saving a single image to disk in the separatte thread took: "+str(took))
            totall_time_spent_saving+=took
            '''
            if experiment_settings_dict["save_probs"]:
                # probabilities are floats scaled by multiplication and converted to uint8
                new_meta["count"] = probs.shape[0]
                new_meta["dtype"] = np.uint8
                #update name to show that it includes probs and not predictions
                path_to_probabilities = Path(path_to_probabilities).parent/("PROBS_"+Path(path_to_probabilities).name)

                with rasterio.open(path_to_probabilities, "w", **new_meta) as dest:
                    dest.write(np.array((probs *255),dtype=np.uint8))
                took=time.time()-saving_data_start_time
                if verbose:print("saving a single image to disk in the separatte thread took: "+str(took))
                totall_time_spent_saving+=took
            elif experiment_settings_dict["save_preds"]:
                preds = np.array(probs.argmax(axis=0),dtype=np.uint8) # since this is a numpy array and not a pytorch array . we now need to use axis=0 instead of dim=0
                new_meta["count"]=1
                new_meta["dtype"]=np.uint8
                with rasterio.open(path_to_probabilities, "w", **new_meta) as dest:
                    dest.write(np.expand_dims(preds,axis=0))
                took=time.time()-saving_data_start_time
                if verbose:print("saving a single image to disk in the separatte thread took: "+str(took))
                totall_time_spent_saving+=took








    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully by terminating the process
        pass

def infer_with_get_preds_on_all(training,files):
    dl = training.learn.dls.test_dl(files)
    #no gradients should be computed during inference!
    with torch.no_grad():
        training.learn.model.eval() #this should not be nececeary but maybe it isnt activated when using a custom backbone?
        training.learn.validate(dl=dl)
    #training.learn.get_preds(dl=dl,with_input=with_input)

    #return the probs for all images in the list files
    #print("preds:"+str(preds))
    #print("len(preds):"+str(len(preds)))
    #return [pred[0] for pred in preds[1] ]


def infer_on_single_image(training,a_file):
    """
    :param training:fastai2 object that have aces to the model and the dataloader
    :param a_file:
    :param experiment_settings_dict:
    :param output_folder:
    :param show:
    :return: dictionary with paths to the created files
    """

    print("classifying : "+str(a_file),end='\r')
    infer_single_image_start= time.time()
    dl = training.learn.dls.test_dl([a_file]) # dl = training.learn.dls.test_dl(all_files) #
    #Does not work with 4 chanel tensors
    #if show:
    #    dl.show_batch()

    #sending with_input=True to get_preds() makes it return the input together with the predictions
    with_input =True

    preds = training.learn.get_preds(dl=dl,with_input=with_input)
    if with_input:
        the_input= preds[0]
        the_prediction =  preds[1][0]
        print("infering on single image took: "+str(time.time()-infer_single_image_start))
        return the_prediction
  

def extend_list_to_multiple_of_x(lst,x):
    """
    use this function in order to make dataset divisible by bachsize
    """
    length = len(lst)
    remainder = length % x
    if remainder != 0:
        items_to_add = x - remainder
        #for i in range(items_to_add):
        extra_items= [lst[i] for i in range(items_to_add)]
        lst.extend(extra_items)
        #lst.extend([lst[-1]] * items_to_add)
    return lst

def infer_all(experiment_settings_dict,benchmark_folder,output_folder,show,all_txt,data_to_save_queue):
    """
    Replacing infer_on_all 
    allowing for  batchwize classification
    allowing for  multiple workers and prefetching of data

    OBS!: I do not get as good results when doing this with model(data_batch)(very fast solution) . Unclear why, but untill I get that to work properly  I use the old solution that predicts image by image


    :param experiment_settings_dict: a dictionary holding the parameters for the trainer that will be used for classification
    :param benchmark_folder: the folder with images that will be clæassified
    :param output_folder: the folder where the resuling predictions will be saved
    :param  data_to_save_queue. A multiprocessing queue to send the data that should be save to a separate process that handles the saving to Disk (otherwise a bottleneck) 


    Saves semantic-segmentation-inference-images in output_folder
    """
    infer_on_all_start_time = time.time()
    list_of_created_files = []


    all_files = Path(all_txt).read_text().split('\n')
    all_files=[Path(benchmark_folder)/Path(experiment_settings_dict["datatypes"][0])/Path(a_path) for a_path in all_files]


    #make sure that all files are of correct type
    im_type= experiment_settings_dict["im_type"]
    all_files=[im_file for im_file in all_files if im_type in im_file.name]
    #in order to infer batch by batch we need to make sure that the length of the dataset can be divided by batchsize
    all_files =extend_list_to_multiple_of_x(all_files,experiment_settings_dict["batch_size"])
    print("####################")
    print("classifying in totall : "+str(len(all_files))+ " nr of images")
    print("segmentation images are saved to: " + str(output_folder))

    print(str(experiment_settings_dict))




    #create a classifier
    dls = sdfi_dataset.get_dataset(experiment_settings_dict)
    training= train.BasicTrainingFastai2(experiment_settings_dict,dls)
    #load saved weights
    training.learn.load(str(pathlib.Path(experiment_settings_dict["model_to_load"]).resolve().with_suffix("")),weights_only=False)

    #classify all images in benchmark_folder
    dl = training.learn.dls.test_dl(all_files,num_workers=experiment_settings_dict["num_workers"]) # dl = training.learn.dls.test_dl(all_files) #


    #ad callback that saves predictions to disk
    save_callback = SaveSegmentationOutput(training.learn, Path(output_folder),np.array(dl.items).reshape(-1,int(experiment_settings_dict["batch_size"])),data_to_save_queue,experiment_settings_dict)
    training.learn.add_cb(save_callback)


    # Move the model to the GPU if available
    if torch.cuda.is_available():
        training.learn.model.cuda()

    #make sure outputfolder exists
    os.makedirs(output_folder, exist_ok=True)



    #sending with_input=True to get_preds() makes it return the input together with the predictions
    # Iterate over all batches in the DataLoader
    batch_inference_loop_start = time.time()
    batch_predictions=[]



    ## using callbacks gave noisy probabiliteis but good argmax() , I must be doing something wrong !###

    #save images in callback 

    infer_with_get_preds_on_all(training,all_files)
    for process in range(int(experiment_settings_dict["save_workers"])):
        data_to_save_queue.put(None) # Each save worker needs a Signal to tell it that we are done
    print("all data has now been predicted")

    return []




def infer_on_all(experiment_settings_dict,benchmark_folder,output_folder,show,all_txt):
    print("OBS! running infer without separate threads for saving output to disk might be much slower than utilizing many threads for saving outut to disk!")
    print("if running inference on a computer with GPU and many cpu cores, consider using a large batchsize and several feeder workers and saving_workers!")
    infer_on_all_start_time =time.time()
    """
    :param experiment_settings_dict: a dictionary holding the parameters for the trainer that will be used for classification
    :param benchmark_folder: the folder with images that will be clæassified 
    :param output_folder: the folder where the resuling predictions will be saved
    :param: show: should the predictions be visualized after classification?
    
    :return: list of paths to the created files

    Saves semantic-segmentation-inference-images in output_folder
    """

    #list paths to all images   
    all_files = Path(all_txt).read_text().split('\n')
    all_files=[Path(benchmark_folder)/Path(experiment_settings_dict["datatypes"][0])/Path(a_path) for a_path in all_files]
    print("####################")
    print("classifying in totall : "+str(len(all_files))+ " nr of images")
    print("segmentation images are saved to: " + str(output_folder))

    list_of_created_files = []
    
    #all_files = [experiment_settings_dict["path_to_dataset"]/Path("images")/Path("")/Path(a_path) for a_path in all_files]
    


    if experiment_settings_dict["dev_mode"]:
        #only classify 10 images if running in debug_mode. shuffle to make sure that we get different images
        random.shuffle(all_files)
        all_files=all_files[-10:]
    
    

    #make sure that all files are of correct type
    im_type= experiment_settings_dict["im_type"]
    all_files=[im_file for im_file in all_files if im_type in im_file.name]
    print(str(experiment_settings_dict))
    

    #create a classifier
    dls = sdfi_dataset.get_dataset(experiment_settings_dict)
    training= train.BasicTrainingFastai2(experiment_settings_dict,dls)
    #load saved weights
    training.learn.load(str(pathlib.Path(experiment_settings_dict["model_to_load"]).resolve().with_suffix("")),weights_only=False)

    #make sure outputfolder exists
    os.makedirs(output_folder, exist_ok=True)




    #classify all images in benchmark_folder
    new_image_start_time = time.time()
    for a_file in all_files:

        with rasterio.open(a_file) as src:
            # make a copy of the geotiff metadata so we can save the prediction/probabilities as the same kind of geotif as the input image
            new_meta = src.meta.copy()
            new_xform = src.transform




        if show:
            #show input image
            tmp_numpy=np.array(Image.open(a_file))
            #nir will be visualized as alpha channel. We remove it to make image visualizable
            tmp_numpy=tmp_numpy[:,:,0:3]
            Image.fromarray(tmp_numpy).show()

        dictionary_with_created_files={}
        probs= infer_on_single_image(training,a_file)
        if experiment_settings_dict["crop_size"]:
            y_index_start =int((probs.shape[1]- experiment_settings_dict["crop_size"])/2)
            x_index_start = int((probs.shape[2]- experiment_settings_dict["crop_size"])/2)
            # create a translation transform to shift the pixel coordinates
            crop_translation = rasterio.Affine.translation(x_index_start, y_index_start)
            # prepend the pixel translation to the original geotiff transform
            new_xform = new_xform * crop_translation
            new_meta['width'] = int(experiment_settings_dict["crop_size"])
            new_meta['height'] = int(experiment_settings_dict["crop_size"])
            new_meta['transform'] = new_xform
            #set the number of channels in the output
            new_meta["count"]=probs.shape[0]


            y_index_end = y_index_start+int(experiment_settings_dict["crop_size"])
            x_index_end = x_index_start+int(experiment_settings_dict["crop_size"])



            probs= probs[:,y_index_start:y_index_end,x_index_start:x_index_end]
        # write the geotiff to disk
        path_to_probabilities = Path(output_folder)/Path("PROBS_"+a_file.name)
        if experiment_settings_dict["save_probs"]:
            time_save_start = time.time()
            if experiment_settings_dict["saved_probs_format"] == "uint8":
                save_probabilities_as_uint8_no_queue(probs=probs, path_to_probabilities=path_to_probabilities,new_meta=new_meta)
            elif experiment_settings_dict["saved_probs_format"] == "float32":
                save_probabilities_as_float32(probs=probs, path_to_probabilities=path_to_probabilities,new_meta=new_meta)
            else:
                sys.exit("no known format to save probs in:"+str(experiment_settings_dict["saved_probs_format"]))
            time_save_end = time.time()
            print("saving the probs took:"+str(time_save_end-time_save_start))


        if experiment_settings_dict["save_preds"]:
            preds = np.array(probs.argmax(dim=0),dtype=np.uint8)

            new_meta["count"]=1
            new_meta["dtype"]=np.uint8
            path_to_predictions = Path(output_folder)/Path(a_file.name)

            with rasterio.open(path_to_predictions, "w", **new_meta) as dest:
                dest.write(np.expand_dims(preds,axis=0))
            dictionary_with_created_files["path_to_predictions"]=path_to_predictions
        print("time for one image is :"+str(time.time()-new_image_start_time))
        new_image_start_time = time.time()
      
      

    print("list_of_created_files:"+str(list_of_created_files))
    print("infer_on_all DONE, result in : "+str(output_folder))

    infer_on_all_end_time =time.time()
    print("infer_on_all took :"+str(infer_on_all_end_time-infer_on_all_start_time))

    return list_of_created_files


def ad_values_nececeary_for_dataset_loader_creation(experiment_settings_dict):
    """
    :param experiment_settings_dict: a dictionary holding the parameters for the trainer that will be used for classification
    return None

    creates 'path_to_dataset' and 'path_to_all_txt' entries in the dictionary by copying the values for 'benchmark_folder' and 'path_to_all_benchmarkset_txt'
    sets 'model_folder' entrie to the parent folder of 'model_to_load'
    TODO! adding these keys and values to the dictionary is a hack, and the need for this should be eliminated. (need to avoid creating a learner)
    """
    experiment_settings_dict["path_to_all_txt"]=experiment_settings_dict["path_to_all_benchmarkset_txt"]
    experiment_settings_dict["path_to_valid_txt"]=False

    #fastai asumes 'model_folder' to be a path that is relative to 'log_folder'. In order to make it relative to the location the script is run from we need to make it absolute with 'resolve()' first.
    experiment_settings_dict["model_folder"]=experiment_settings_dict["model_to_load"].parent.resolve()
    experiment_settings_dict["log_folder"]=(experiment_settings_dict["model_to_load"].parent.parent/Path("logs"))
    experiment_settings_dict["sceduler"]= "fit_one_cycle" #this should really ot be needed but is demanded by the code for now

def main(config):

    if not torch.multiprocessing.get_start_method(allow_none=True):
        #this is nly alowed to be set once. and the if statement above makes sure its only set once
        torch.multiprocessing.set_start_method('spawn') # otherwise we get : Error      "Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spa>



    # Check if CUDA is available
    print("##########################################")
    if torch.cuda.is_available():
        # Get the name of the current GPU device
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"PyTorch is using GPU: {device_name}")
    else:
        print("PyTorch is using CPU")
    print("##########################################")

    experiment_settings_dict= sdfi_utils.load_settings_from_config_file(config)
    ad_values_nececeary_for_dataset_loader_creation(experiment_settings_dict)

    benchmark_folder = experiment_settings_dict["benchmark_folder"]
    if "output_folder" in experiment_settings_dict:
        output_folder= experiment_settings_dict["output_folder"]
    else:
        output_folder = experiment_settings_dict["model_to_load"].parent/pathlib.Path(pathlib.Path(benchmark_folder).parent.parent.name)




    show= "show" in experiment_settings_dict and experiment_settings_dict["show"] # False #debug variable , show input and output of inference


    #Default configuration is to use separate processes for loading data(only works on linux) saving data (works on both platforms) and inference (can be done on GPU)
    #running everything in the same process might be faster if the dataset is very small (skipping overhead for creating and closing down processes and queues )or if there is no GPU available and not enough cpu cores to infer save and load in separate processes on separate cores
    #running everything in the same process might also be more cpu-hours efficient
    if "save_workers" in experiment_settings_dict and int(experiment_settings_dict["save_workers"])>0:
        #using a set of workers for saving the predictions to disk
        # Create a multiprocessing queue for sending prediction_probabilities_images to be saved to disk as images
        #set a max size. if this is reached we need to have more workers that handle the data in the queue
        queue = Queue(20)
        # Create processes for saving prediction_probabilites as images (one proces for each save_worker 
        save_inference_proces_workers = [ Process(target=save_probabilities_as_uint8, args=(queue,)) for process in range(int(experiment_settings_dict["save_workers"]))]
            
        
        # Create a process for doing inference
        inference_process = Process(target=infer_all, args=(experiment_settings_dict,benchmark_folder,output_folder,show,experiment_settings_dict["path_to_all_benchmarkset_txt"], queue))

        try:
            # Start the processes 
            for proces in save_inference_proces_workers:
                proces.start()
            inference_process.start()

            # Wait for the inference proces to finnish
            inference_process.join()

            # Wait for the save_inference_proces_workers to finish
            for proces in save_inference_proces_workers:
                proces.join()

        except KeyboardInterrupt:
            # Terminate both processes if Ctrl+C is pressed during the execution
            inference_process.terminate()
            for process in save_inference_proces_workers:
                process.terminate()
        finally:
            # Close the queue
            queue.close()
            queue.join_thread()

    else:
        infer_on_all(experiment_settings_dict=experiment_settings_dict, benchmark_folder=benchmark_folder,
                     output_folder=output_folder, show=show,
                     all_txt=experiment_settings_dict["path_to_all_benchmarkset_txt"])



    ##infer_all(experiment_settings_dict=experiment_settings_dict,benchmark_folder = benchmark_folder,output_folder=output_folder,show=show,all_txt=experiment_settings_dict["path_to_all_benchmarkset_txt"])
    ##

    print("DONE processing all images in :"+str(benchmark_folder))

if __name__ == "__main__":
    """
    Do inference on all images, given one or more dictionaries that defines an inference job (e.g what images to do semantic segmentation on , and what  model to use)

    """
    usage_example="example usage: \n "+r"python infer.py --config ./configs/example_inference.ini"
    # Initialize parser
    parser = argparse.ArgumentParser(
                                    epilog=usage_example,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument("-c", "--config", help="one or more paths to experiment config file",nargs ='+',required=True)
    args = parser.parse_args()

    for config in args.config:
        main(config)


        
        
