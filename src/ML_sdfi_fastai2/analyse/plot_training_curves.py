import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import os


#python plot_training_curves.py --LogFile \mnt\logs\RoofTopRGB1000p_Byg_Iteration27\esbjerg_1000x1000_oversampled_3.csv \home\rasmusjohansson\mortens_iteration_27_visualised.xlsx




def create_summary(log_files, output_sumary_file):
    pandas_dataframes=[]

    #READ IN THE TRAINING LOGS
    for log_file in log_files:
        file_name=os.path.split(pathlib.Path(log_file))[-1].split(".")[-2]


        #assert ".csv" in log_file, print("the log files should be of type  xxx.csv ")
        if ".csv" in str(log_file):
            pandas_dataframe = pd.read_csv(log_file)
        else:
            pandas_dataframe = pd.read_excel(log_file)

        for column_to_drop in ['AntalPixler', 'time']:
            try:
                pandas_dataframe = pandas_dataframe.drop(columns=[column_to_drop])
            except:
                print("could not drop : "+str(column_to_drop))
        #pandas_dataframe = pandas_dataframe.set_index("epoch")
        #rename the column names to show from what file each column originates from before merging
        pandas_dataframe= pandas_dataframe.rename(columns=lambda originalname: file_name+originalname)


        pandas_dataframes.append(pandas_dataframe)

    print(pandas_dataframes)




    joined_pandas_dataframes = pd.concat(pandas_dataframes, axis=1, join="outer")


    with open(output_sumary_file, 'w') as out_file:
        out_file.writelines(["##############################################################################","\n","################################MIN VALUES####################################","\n",str(joined_pandas_dataframes.min()),"\n","##############################################################################"])
        #out_file.write("################################MIN VALUES####################################")
        #out_file.write(str(joined_pandas_dataframes.min()))
        #out_file.write("##############################################################################")

def get_value(pandas_dataframe,column_name,value_in_name):
    """
    Get the min or max value for  certain column
    """
    if value_in_name =="min":
        return pandas_dataframe[column_name].min()
    elif value_in_name =="max":
        return pandas_dataframe[column_name].max()
    else:
        sys.exit("value_in_name should be 'min' or 'max'")
    


def create_plot(log_files, output_plot_file, use_log_format,values_to_plot,ylim,image_format="png",show=False,value_in_name="min"):
    plt.close("all")
    

    pandas_dataframes = []
    styles = {}  # }{"train_loss","dashed"}




    #READ IN THE TRAINING LOGS
    for log_file in log_files:
        file_name=os.path.split(pathlib.Path(log_file))[-1].split(".")[-2]


        #assert ".csv" in log_file, print("the log files should be of type  xxx.csv ")
        if ".csv" in str(log_file):
            pandas_dataframe = pd.read_csv(log_file)
        else:
            pandas_dataframe = pd.read_excel(log_file)
        print(pandas_dataframe)
        print(values_to_plot)


        pandas_dataframe_selected_data = pandas_dataframe[values_to_plot]
        print(pandas_dataframe_selected_data)
        #pandas_dataframe_selected_data=pandas_dataframe_selected_data.set_index("epoch")
        # rename the column names to show from what file each column originates from before merging
        

        pandas_dataframe_selected_data = pandas_dataframe_selected_data.rename(columns=lambda originalname: file_name + originalname+ value_in_name+": "+str(get_value(pandas_dataframe_selected_data,originalname,value_in_name)) )
        #input(pandas_dataframe_selected_data["nonfrozen_esbjerg_1000x1000erlytrain_loss"].min())


        #adjust how the different lines look
        styles[file_name+"train_loss"]='--'
        styles[file_name + "IoU_Baggrund"] = '--'
        #pandas_dataframe.plot(ylim=[0, 0.1],style=styles)



        pandas_dataframes.append(pandas_dataframe_selected_data)





    joined_pandas_dataframes = pd.concat(pandas_dataframes, axis=1, join="outer")
    joined_pandas_dataframes.plot(ylim=ylim,style=styles,logy=use_log_format)

    plt.subplots_adjust(bottom=0.6) #making space for the legend on the right side
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -1.6))  # place legend outside of plot
    plt.savefig(fname=output_plot_file, format=image_format)
    if show:
        print("showing plot")
        plt.show()

    plt.close("all")






if __name__ == "__main__":
    # Initialize parser
    # python forretningsrapport.py --help to get more information on usage
    usage_example = "python plot_training_curves.py - f \mnt\logs\fixed_lr\fixed_lr.csv \mnt\logs\nonfrozen_esbjerg_1000x1000\nonfrozen_esbjerg_1000x1000.csv - o fixedVsfitonecykle"
    parser = argparse.ArgumentParser(
        epilog=usage_example,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-f', '--LogFile', nargs='+',
                        help='one or more log.csv files  containing valid and training loss for each epoch',
                        required=True)
    parser.add_argument('-o', '--OutputPrefix', help='prefix for plot names, e.g myprefix -> myprefix_newVsOld.png ',
                        required=True)
    parser.add_argument('-s', '--Show', help='should we show the image before saving to disk', required=False,
                        dest='show', action='store_true')
    parser.add_argument('-l', '--PlotLog', help='should we plot in log format? ', required=False, dest='plotlog',
                        action='store_true')
    parser.add_argument('-y', '--ymax', help='max value on y axle ', required=False, default = 1.0,type =float)
    parser.add_argument('-m', '--min_or_max', help='should we include min or max values in line names? ', required=False, default = "min")






    parser.set_defaults(show=False)

    import matplotlib.pyplot as plt

    plt.close("all")

    args = parser.parse_args()
    """
    usage: 
    python plot_training_curves.py --LogFile \mnt\logs\RoofTopRGB1000p_Byg_Iteration27\esbjerg_1000x1000_oversampled_3.csv \home\rasmusjohansson\mortensfavoritmodel.xlsx \mnt\logs\r34-esbjergplus_oversampledindustri_blok1000p_rasmustest2\oversampledStandardTransform.csv --ymax 0.1
    """



    image_format="png"

    create_summary(log_files=args.LogFile, output_sumary_file=args.OutputPrefix+"sumary.txt")
    values_to_plot=['train_loss', 'valid_loss']
    #values_to_plot = ["valid_accuracy"]
    create_plot(log_files=args.LogFile,  output_plot_file=args.OutputPrefix+"plot."+image_format,values_to_plot=values_to_plot, image_format=image_format,use_log_format=args.plotlog,ylim=[0, args.ymax],show=args.show,value_in_name=args.min_or_max)
