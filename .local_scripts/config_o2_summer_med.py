import math
from time import time
import os
from datetime import datetime, timedelta
start_date="2024-05-31" #%Y-%m-%d
target_prediction_time=89
variable="o2"
basin="med"
n_train=366
device="cpu" #"cpu" or "cpu"


perc=0.95# this is the target coverage probability of the prediction interval

###Do not change this part
dir_path = os.path.dirname(os.path.realpath(__file__))
download_path="/g100_scratch/userexternal/gpadula0/MedScratch/"+variable+"/"+basin+"/{}_{}_{}_{}_{}".format(variable,basin,start_date,n_train,target_prediction_time)

download_re_path="/g100_scratch/userexternal/gpadula0/MedScratch/"+variable+"/"+basin
download_re_path_all="/g100_scratch/userexternal/gpadula0/ClimaScratch/"+variable
plot_path=dir_path+"/plot/"+variable+"/"+basin+"/{}_{}_{}_{}_{}".format(variable,basin,start_date,n_train,target_prediction_time)