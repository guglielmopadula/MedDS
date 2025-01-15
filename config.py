import math
from time import time
import os
from datetime import datetime, timedelta
start_date="2024-01-13" #%Y-%m-%d
target_prediction_time=10
variable="chl"
basin="ion2"
n_train=min(730,max(2*target_prediction_time,120))
timestep=target_prediction_time #it must hold that timestep<target_prediction_time, empirically the performance is better when they are the same
device="gpu" #"gpu" or "cpu"
#n_train=100 #override if you preder

perc=0.95# this is the target coverage probability of the prediction interval

###Do not change this part
current_date = datetime.now().date()
prev_date = current_date - timedelta(days=729)
diff=datetime.strptime(start_date, "%Y-%m-%d").date()-prev_date
max_n_train=diff.days-1
n_train=min(max_n_train,n_train)
del diff, prev_date, current_date, max_n_train 
dir_path = os.path.dirname(os.path.realpath(__file__))
download_path=dir_path+"/"+variable+"/"+basin+"/{}_{}_{}_{}_{}_{}".format(variable,basin,start_date,n_train,target_prediction_time,timestep)
