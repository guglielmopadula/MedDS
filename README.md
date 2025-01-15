# Digital Shadow of the Mediterrean Copernicus Analysis and Forecast System
This is the repository that containts the code of the Digital Shadow of the Mediterrean Copernicus Analysis and Forecast System which is part of the projects developed under the Research Topic 4 of the Spoke 9 of the INEST project, as a joint work between SISSA and OGS.

Let's explore how to use it

# First step: configure it
In the main directory, there if a file called config.py
In this file there are several variables
```python 
start_date="2024-01-13" #%Y-%m-%d
```

This indicates the date in which the last piece of data is available.

```python 
target_prediction_time=10
```

This indicates how many days do you want to do predictions.

```python 
variable="chl"
```

This indicates the variable we want to predict. The full list is in staticvariables.py.

```python 
basin="ion2"
```

```python 
n_train=min(730,max(2*target_prediction_time,120))
```
This is the number of training data that do you want to use. It is reccomended to leave that as it it.

```python 
timestep=target_prediction_time 
```
This indicates the number of timesteps that the model does in each iteration. It is reccomended to leave it as it is.
```python
device="gpu" #"gpu" or "cpu"
```
Only cuda is currently supported.

```python
perc=0.95
```
this is the target coverage probability of the prediction interval

## Second step: download data
Do
```bash
python download_data.py config
```
to download data. A Copernicusmarine account is required.
Optionally, download the test data (data for the predictions you want to perform), if available
```bash
python download_test_data.py config
```

## Third step: compute the predictions

Then compute the predictions
```bash
python compute_predictions.py config
```
and plot them
```bash
python plot_predictions.py config
```
## Fourth step: validate
To validate, you need to have downloaded the test data.
The command
```bash
python theoretical_validation.py config
```
checks if the test data is consistent. The procedure is in the attached pdf.
```bash
python practical_validation.py config
```
does some plots, and computes biases and rmsds in a csv file.

## How to update
To perform data assimilation, do a
```bash
python update_data.py config
```
followed by a 
```bash
python compute_predictions.py config
```
