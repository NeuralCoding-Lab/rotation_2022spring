import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import matplotlib.pyplot as plt
import os, sys,glob, copy
import scipy.sparse as sp
import xarray as xr
from scipy.ndimage.filters import gaussian_filter
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_session import (
    EcephysSession, 
    removed_unused_stimulus_presentation_columns
)
from allensdk.brain_observatory.ecephys.visualization import plot_mean_waveforms, plot_spike_counts, raster_plot
from allensdk.brain_observatory.visualization import plot_running_speed


def nanunique(x):
    """np.unique get rid of nan numbers."""
    temp = np.unique(x.astype(float))
    return temp[~np.isnan(temp)]

def save_npz(matrix, filename):
    matrix_2d = matrix.reshape(matrix.shape[0], int(len(matrix.flatten())/matrix.shape[0]))
    sparse_matrix = sp.csc_matrix(matrix_2d)
    np.savez(filename, [sparse_matrix, matrix.shape])
    return 'npz file saved'

def load_npz(filename):
    """
    load npz files with sparse matrix and dimension
    output dense matrix with the correct dim
    """
    npzfile = np.load(filename, allow_pickle=True) 
    sparse_matrix = npzfile['arr_0'][0]
    ndim=npzfile['arr_0'][1]

    new_matrix_2d = np.array(sparse_matrix.todense())
    new_matrix = new_matrix_2d.reshape(ndim)
    return new_matrix




####################################################################################################################
# get session
pd.set_option("display.max_columns", None)
mouseID = 'mouse421529'
key =  'natural_scenes'
#brain observatory
condi='cortex_nwb2'

#output = '/local1/work_allen/Ephys/resorted/'+mouseID
output = 'C:/Users/Neural Coding/Desktop/Jia-lab/data/resorted/'+mouseID
output_path = output+'/matrix/'
output_meta = mouseID+'_'+condi+'_meta.csv'


session_id = 715093703
mouse_id = '421529'
basepath = "C:/Users/Neural Coding/Desktop/jia-lab/data/ecephys_cache_dir/"
manifest_path = os.path.join(basepath, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

session = cache.get_session_data(session_id,
                                 amplitude_cutoff_maximum=np.inf,
                                 presence_ratio_minimum=-np.inf,
                                 isi_violations_maximum=np.inf)

# all units in one session
# calculate response features
output = 'C:/Users/Neural Coding/Desktop/jia-lab/data/resorted/'+mouseID
output_path = output+'/matrix/'
output_meta = mouseID+'_cortex_nwb2_meta.csv'

f = output_path+output_meta

df = session.units  # 2073 units
df = df.rename(columns={"channel_local_index": "channel_id", 
                        "ecephys_structure_acronym": "ccf", 
                        "probe_id":"probe_global_id", 
                        "probe_description":"probe_id",
                       'probe_vertical_position': "ypos"})
df['unit_id']=df.index
# channel_id is not strictly sorted, but roughly sorted according to depth

# get stim_table
stim_table = session.get_stimulus_table([key])
stim_table=stim_table.rename(columns={"start_time": "Start", "stop_time": "End"})
duration = round(np.mean(stim_table.duration.values), 2)
print(duration)
stim_table.to_csv(output_path+'stim_table_'+key+'.csv')  # make stim table file

#binarize tensor
# binarize with 1 second bins
time_bin_edges = np.linspace(0, duration, int(duration*1000)+1)

# select cortex
cortex = ['VISp', 'VISl', 'VISli', 'VISrl', 'VISal', 'VISam', 'VISpm']
cortical_units_ids = np.array([idx for idx, ccf in enumerate(df.ccf.values) if ccf in cortex])
print(len(cortical_units_ids))

# get binarized tensor
df_cortex = df.iloc[cortical_units_ids]

spike_counts_da = session.presentationwise_spike_counts(
    bin_edges=time_bin_edges,
    stimulus_presentation_ids=stim_table.index.values,
    unit_ids=df_cortex.unit_id.values  # select cortex unit_ids 2073 --> 637,'VISp', 'VISl', 'VISli', 'VISrl', 'VISal', 'VISam', 'VISpm'
)

matrix = spike_counts_da.values
matrix = np.rollaxis(matrix, -1,0)

save_npz(matrix, output_path+key+'_'+condi+'.npz')


# print(matrix.shape)
assert len(df_cortex)==matrix.shape[0]
mean_spike_count = list(matrix.mean(1).mean(1))


############################################### try top 10% responsive units(60) ###############################################
top60_units = np.argsort(-mean_spike_count)[0:60]
df_cortex1 = df_cortex ; df_cortex1.index = range(637)
top_df_cortex = df_cortex1.iloc[top60_units,:]
top_spike_counts_da = session.presentationwise_spike_counts(
    bin_edges=time_bin_edges,
    stimulus_presentation_ids=stim_table.index.values,
    unit_ids=top_df_cortex.unit_id.values )
top_matrix = top_spike_counts_da.values
top_matrix = np.rollaxis(top_matrix, -1,0)
# draw
plt.figure()
plt.plot(top_matrix.mean(1).mean(1))
plt.xlabel(u"responsive units") 
plt.ylabel("mean spike count(ms)") 
# plt.title("A simple plot") 
plt.show()

top_spikes =  np.zeros((np.shape(top_matrix)[0], len(stim_table['frame'].unique()), rep, np.shape(top_matrix)[2])).astype('uint8')
for idx_i, i in enumerate(images):
    tmp = top_matrix[:,np.where((stim_table['frame']==i))[0],:]
    top_spikes[:,idx_i,:,:] = tmp.astype('uint8')
    
top_firing_rate =  (top_spikes[:,:,:,1:250].sum(3)/float(250)).mean(2).mean(0) # FR in 1ms 



# another method for Cal FR
df_cortex['FR']=(matrix[:,:,50:250].sum(2)/float(200)).mean(1)   # 637 units, get firing rate, ignore 0-50 ms
# df_cortex.to_csv(f, encoding='utf-8')



# load required files
matrix_unit_binarized = load_npz(output_path+key+'_'+condi+'.npz')
stim_table = pd.read_csv(output_path+'stim_table_'+key+'.csv')
condi='cortex_nwb2'
# repeats for all conditions
rep = int(np.shape(matrix_unit_binarized)[1]/(len(stim_table['frame'].unique())))
# rep=50
spikes = np.zeros((np.shape(matrix_unit_binarized)[0], len(stim_table['frame'].unique()), rep, np.shape(matrix_unit_binarized)[2])).astype('uint8')
images = np.sort(stim_table['frame'].unique())
for idx_i, i in enumerate(images):
    tmp = matrix_unit_binarized[:,np.where((stim_table['frame']==i))[0],:]
    spikes[:,idx_i,:,:] = tmp.astype('uint8')

save_npz(np.squeeze(spikes), 'C:/Users/Neural Coding/Desktop/Jia-lab/CORnet/data/'+ 'image_'+condi+'.npz')


matrix = load_npz(output_path+key+'_'+condi+'.npz') # load npz
# spikes = load_npz(output_path+key+'_'+condi+'.npz') # cannot load_npz directly
# Cal FR
firing_rate =  (spikes[:,:,:,0:250].sum(3)/float(250)).mean(2).mean(0) # 

##################################################### Using CORnet ###############################################
# load different layers of CORnet output feature
CORnetS_feats = np.load("C:\\Users\\Neural Coding\\Desktop\\Jia-lab\\CORnet\\CORnet-S_V4_output_feats.npy" , mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
CORnetS_feats2  = np.load("C:\\Users\\Neural Coding\\Desktop\\Jia-lab\\CORnet\\CORnet-S_V2_output_feats.npy" , mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

# 1. try classification
import torch as t
from torch import nn
from torch.nn import functional as F

connected_layer = nn.Linear(CORnetS_feats.shape[1], 20) # try to divide into 20 categories
output = connected_layer(t.Tensor(CORnetS_feats))
b=F.softmax(output,dim=1)
c = t.max(b, 1)[1].numpy()

# try PCA
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
pca.fit(CORnetS_feats)
print (pca.explained_variance_ratio_)
print( pca.explained_variance_)
print (pca.n_components_)

# X_new = pca.transform(CORnetS_feats)
# plt.scatter(X_new,marker='o')
# plt.show()

# try RDM
import rsatoolbox
data = rsatoolbox.data.Dataset(CORnetS_feats)
rdms = rsatoolbox.rdm.calc_rdm(data)
rsatoolbox.vis.show_rdm(rdms)

firing_rate3 = firing_rate1.mean(0)
data = rsatoolbox.data.Dataset(firing_rate3)
rdms = rsatoolbox.rdm.calc_rdm(data)
rsatoolbox.vis.show_rdm(rdms)

# 2. Using CORnet with linear regerssion to predict neural activity
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import explained_variance_score

# define training and test set
X = CORnetS_feats2
Y = np.array(firing_rate[1:119]) # image id: -1 (as contrast)
top_Y = np.array(top_firing_rate[1:119]) 

# train model
regr = linear_model.LinearRegression()

# cross validtion: leave one out
def _leave_one_out(algr, X, y,pred_y):
    loo = LeaveOneOut()
    square_error_sum = 0.0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = algr.fit(X_train, y_train.ravel())
        predicted_y = model.predict(X_test)
        square_error_sum += float(y_test[0] - predicted_y) ** 2
        pred_y.append(predicted_y)
    mse = square_error_sum / X.shape[0]
    print ('-----------------------')
    print ('Leave One Out?mse ' , mse)
    print ('-----------------------')


pred_y = []
_leave_one_out(regr, X, top_Y,pred_y)

# top 60 responsive units
pred_y = []
_leave_one_out(regr, X,top_Y,pred_y)
print('Explained Variance score: %.3f' % explained_variance_score(top_Y, pred_y ) )

# plot FR comparison
plt.plot(names, Y, color = 'black', mec='r', mfc='w',label=u'true')
plt.plot(names, pred_y, color = 'red', ms=10,label=u'predict')
plt.legend()  
# plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"images") 
plt.ylabel("Firing rate") 
# plt.title("A simple plot") 
plt.show()

# model evaluation
print('intercept_:%.3f' % regr.intercept_)
print('coef_:', regr.coef_)
print('Mean squared error: %.3f' % mean_squared_error(Y,pred_y)) ##((y_test-regr.predict(x_test))**2).mean()
print('Variance score: %.3f' % r2_score(Y, pred_y ) )
print('Explained Variance score: %.3f' % explained_variance_score(Y, pred_y ) )






























