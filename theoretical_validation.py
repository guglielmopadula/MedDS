import numpy as np
from scipy.stats import truncnorm, binom
import netCDF4 as nc
import sys
config=__import__(sys.argv[1])

test_output_filename="test_{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.target_prediction_time)
test_dataset=nc.Dataset(config.download_path+"/"+test_output_filename+".nc")

test_dataset=test_dataset.variables[config.variable][:]
testing_data=test_dataset.data[np.logical_not(test_dataset.mask)].reshape(config.target_prediction_time,-1)
Winv=np.load(config.download_path+"/matrix.npy")
data_U_test=np.load(config.download_path+"/latent_mean.npy")
std_U_test=np.load(config.download_path+"/latent_std.npy")

lat_std_vecs=np.load(config.download_path+"/latent_std_vecs.npy")
U_test=testing_data@Winv
rank=U_test.shape[1]
lb_qf=binom.ppf(q=0.005,n=rank,p=0.95)/rank
ub_qf=binom.ppf(q=0.995,n=rank,p=0.95)/rank


value=np.mean(np.logical_and(U_test<data_U_test+1.96*np.sqrt(std_U_test**2+lat_std_vecs**2),U_test>data_U_test-1.96*np.sqrt(std_U_test**2+lat_std_vecs**2)))
if lb_qf<=value<=ub_qf:
    print("Model is validated")

else:
    print("Model is not validated")