# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 20:21:32 2022

@author: zhuyue1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('default')
get_ipython().run_line_magic('matplotlib', 'inline')

def load_npz(filename):
    npzfile = np.load(filename, allow_pickle=True, encoding='latin1') 
    sparse_matrix = npzfile['arr_0'][0]
    ndim = npzfile['arr_0'][1]

    new_matrix_2d = np.array(sparse_matrix.todense())
    new_matrix = new_matrix_2d.reshape(ndim)
    return new_matrix

# load preprocessed matrix: neuron*condition*time (binsize=1ms) binarized matrix with 0 and 1
matrix = load_npz(filename = 'E:/rotation_Jialab/JiaXX lab/rotation_data/drifting_gratings_cortex_nwb2.npz')
# load metadata for stimulus
stim_table = pd.read_csv('E:/rotation_Jialab/JiaXX lab/rotation_data/stim_table_drifting_gratings.csv', index_col=0)


# example code to organize stimulus condition into repeats for each condition
oris = np.sort(stim_table.orientation.unique())[:-1]
tfs = np.sort(stim_table.temporal_frequency.unique())[:-1]

rep = int(np.shape(matrix)[1]/(len(oris)*len(tfs)))
spikes = np.zeros((np.shape(matrix)[0], len(oris), len(tfs), int(rep), np.shape(matrix)[2])).astype('uint8')

for idx_o, o in enumerate(oris):
    for idx_t, t in enumerate(tfs):
        tmp = matrix[:,np.where((stim_table['orientation']==o) & (stim_table['temporal_frequency']==t))[0],:]
        # pad 0 if there is missing rep
        if np.shape(tmp)[1]<np.shape(spikes)[-2]:
            result = np.zeros((np.shape(spikes)[0], np.shape(spikes)[-2], np.shape(spikes)[-1]))
            result[:,:tmp.shape[1],:] = tmp.astype('uint8')
            spikes[:,idx_o,idx_t,:,:] = result.astype('uint8')
        else:
            spikes[:,idx_o,idx_t,:,:] = tmp.astype('uint8')
spikes = np.squeeze(spikes)

# matrix with condition organized by repeats: neuron*ori*tf*rep*time bins
matrix_rep = load_npz('E:/rotation_Jialab/JiaXX lab/rotation_data/drifting_grating_cortex_nwb2_rep.npz')
matrix_rep.shape



##getting the array of single neurons's firing rate under differ drifting grating
n1_F_all = [] 
n1_se_all = []

for t in range(len(tfs)):
    #print(n1_TF_t) 很难实现打印变量名以及对迭代生成的变量名的打印，还是直接存到一个数组里面
    for o in range(len(oris)):   
        # step1 取神经元-1，在方向o,频率t的drifting_grating，下所有时间内的所有平行组重复实验数据 。
        n1 = matrix_rep[3,o,t,0:rep,0:2000]  
        #step2 分别统计每组重复实验所有时间内的spikes counts。
        n1_15rep_sum = np.sum(n1, axis = 1)
        #step3.1 对重复实验组取平均,得到n1 神经元在第o个ori 的最小频率的drifting_grating 下2000ms 内总的spike的平均水平
        #n1_m = np.mean(n1_15rep_sum)
        #step3.2 计算重复组每组 firing rate: spikes counts sum/ time 以及 standard deviation
        n1_15rep_F = n1_15rep_sum/2
        n1_se = np.std(n1_15rep_F)/pow(rep,0.5)
        #step4 计算mean firing_rate 
        n1_F = np.mean(n1_15rep_F) # matrix_rep.ndim[3]  #这个怎么表示？？？？
        #step5 存入array
        n1_F_all.append(n1_F)
        n1_se_all.append(n1_se)
        
#step8 数组升维，change to len(tfs) *len(oris）shape np.array    
n1_F_all = np.array(n1_F_all)        
n1_F_all = n1_F_all.reshape(len(tfs),len(oris))
n1_se_all = np.array(n1_se_all)  #验证std计算的对不对?是对的，因为std 比较大，以坐标为刻度显得线很长。
n1_se_all = n1_se_all.reshape(len(tfs), len(oris))                           

  #neurons 迭代
  #可以试试 def 封装的形式生成firing rate （截至4/8 还没实现）
  #将两幅图画在一张图中，某个点处，把15个rep画出来
  #画一个measuring and interpreting...文献中的fig_6



# tuning curve
for t in range(len(tfs)):
    y = n1_F_all[t,:]
    x = oris
    yerr = n1_se_all[t,:]
    
    plt.xlabel('Orientation (degrees)')
    plt.ylabel('Response (spikes per second)')
    plt.suptitle('Orientation Tuning curve of neuron 4 under {} Hz drifting grating'.format(tfs[t]))
    
    #plt.ylim(1.0, 5) #y轴上下界
    plt.errorbar(x,y,yerr,fmt='o-',ecolor='grey',color='grey',elinewidth=1,capsize=0.5)
    #plt.plot(x, y)
    plt.savefig('E:/rotation_Jialab/neuron4_orientation_tuningcurve{}'.format(t+1))
    #plt.figure(figsize = (3.5,2.88)) #调整画布大小
    plt.clf() #clean canvas,critical for plotting in different pictures.
    
    

  
   
    
# heatmap
tfs_Hz = []
for t in range(len(tfs)):
    tfs_Hz.append('{}Hz'.format(tfs[t])) 
    
'''
# another way: using ‘+’，but u have to convert the float type into str!!!!
tfs1 = tfs.astype(str)
tfs_Hz = []    
for t in range(len(tfs)):
    tfs_Hz.append(tfs1[t]+'Hz')   
    
'''

oris_deg = []
for o in range(len(oris)):
    oris_deg.append('{}deg'.format(oris[o]))
    
fig, ax = plt.subplots()
im = ax.imshow(n1_F_all)

ax.set_xticks(np.arange(len(oris_deg)))
ax.set_yticks(np.arange(len(tfs_Hz)))

ax.set_xticklabels(oris_deg)
ax.set_yticklabels(tfs_Hz)


plt.suptitle('neuron 1 average firing rate in differ conditions of drifting grating temperol frequences and orientations')
plt.setp(ax.get_xticklabels(), rotation = 45, ha="right",rotation_mode = "anchor")    
    
    
    
    
# heatmap with colorbar and gradient ramp 
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts    
    
fig, ax = plt.subplots()

im, cbar = heatmap(n1_F_all, tfs_Hz, oris_deg, ax=ax,
                   cmap="YlGn", cbarlabel="firing rate [spikes/sec]")
texts = annotate_heatmap(im)

fig.tight_layout()

plt.suptitle('neuron 2 average firing rate in differ conditions of drifting grating temperol frequences and orientations')

plt.show()       
    
    
    
    
# seaborn_heatmap
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")
# 绘制x-y-z的热力图，比如 年-月-销量 的热力图
f, ax = plt.subplots(figsize=(9, 6))
#使用不同的颜色
sns.heatmap(flights, fmt="d",cmap='YlGnBu', ax=ax)
#设置坐标字体方向
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')
plt.show()
    
    
## noise correlation    
for o in range(len(oris)):
    # step1 取神经元-1和-2，在方向o,频率t的drifting_grating，下所有时间内的所有平行组重复实验数据 。
    n1 = matrix_rep[0,o,0,0:rep,0:2000]  
    n2 = matrix_rep[1,o,0,0:rep,0:2000]
    
    #step2 分别统计每组重复实验所有时间内的spikes counts。
    n1_15rep_sum = np.sum(n1, axis = 1)
    n2_15rep_sum = np.sum(n2, axis = 1)
    
    #step3.2 计算重复组每组 firing rate: spikes counts sum/ time 
    n1_15rep_F = n1_15rep_sum/2
    n2_15rep_F = n2_15rep_sum/2
    
    #计算noise correlation
    rsc_12 = np.corrcoef(n1_15rep_F,n2_15rep_F)
    rsc_12 = round(rsc_12[0,1],2)
    print(rsc_12)
    
    #plotting
    x = n1_15rep_F
    y = n2_15rep_F
    
    plt.xlabel('Response cell 1 (spikes per s)')    
    plt.ylabel('Response cell 2 (spikes per s)')
    plt.suptitle('Spike count correlation(1Hz, {}deg)'.format(oris[o]))
    
    style = dict(size=10, color='gray')
    plt.text(8, 30, "rsc={}".format(rsc_12), **style)
    
    plt.xlim(-1,12)
    plt.ylim(0,45)
    
    plt.plot(x, y ,'o', color = 'black')
    
    plt.savefig('E:/rotation_Jialab/Spike_count_correlation@{}'.format(o+1)) #why this doesn't work? plt.savefig('E:/rotation_Jialab/Spike count correlation({}deg)'.format(oris[o]))
    
    plt.clf() 





##signal correlation
n1_F_all = []
n2_F_all = []
for o in range(len(oris)):
    # step1 取神经元-1和-2，在方向o,频率t的drifting_grating，下所有时间内的所有平行组重复实验数据 。
    n1 = matrix_rep[0,o,4,0:rep,0:2000]  
    n2 = matrix_rep[1,o,4,0:rep,0:2000]
    
    #step2 分别统计每组重复实验所有时间内的spikes counts。
    n1_15rep_sum = np.sum(n1, axis = 1)
    n2_15rep_sum = np.sum(n2, axis = 1)
    
    #step3 计算重复组每组 firing rate: spikes counts sum/ time 
    n1_15rep_F = n1_15rep_sum/2
    n2_15rep_F = n2_15rep_sum/2
    
    #取平均
    n1_F = np.mean(n1_15rep_sum)
    n2_F = np.mean(n2_15rep_sum)
    
    n1_F_all.append(n1_F)
    n2_F_all.append(n2_F)
    
#计算correlation
rsc_12 = np.corrcoef(n1_F_all,n2_F_all)
rsc_12 = round(rsc_12[0,1],2)
print(rsc_12)
  
#plotting
x = n1_F_all
y = n2_F_all

plt.xlabel('Response cell 1 (spikes per s)')    
plt.ylabel('Response cell 2 (spikes per s)')
plt.suptitle('Signal correlation(15Hz, differ deg)')

style = dict(size=10, color='gray')
plt.text(5, 40, "rsc={}".format(rsc_12), **style)

#plt.xlim(-1,12)
#plt.ylim(0,45)

plt.plot(x, y ,'o', color = 'black')

plt.savefig('E:/rotation_Jialab/Signal correlation(15Hz, differ deg)') 
            #why this doesn't work? plt.savefig('E:/rotation_Jialab/Spike count correlation({}deg)'.format(oris[o]))

#plt.clf() 




  

#加一行将 tfs 转换成字符串
# 每一层的Rsc怎么计算的？
# depth/layer 数据怎么取出来 :根据老师发的文章的示意图，layer是均等分布的，取出所有v1d 数据，把上下限相减得到总厚度
# 3/24 画跟脑区有关的图; 画 normaliza（Z-score）之后的 noise correlation; 4/8 还没有做



'isolation_distance'
'anterior_posterior_ccf_coordinate'
'probe_horizontal_position'
'location'


#layer dependent temporal frequency firing rate

v1 = df[['ccf']=='VISp']
l1 = v1.loc[(v1['dorsal_ventral_ccf_coordinate'] <=810) & (v1['dorsal_ventral_ccf_coordinate'] >=656)]
l23 = v1.loc[(v1['dorsal_ventral_ccf_coordinate'] >=810) & (v1['dorsal_ventral_ccf_coordinate'] <=964)]
l4 = v1.loc[(v1['dorsal_ventral_ccf_coordinate'] >=964) & (v1['dorsal_ventral_ccf_coordinate'] <=1117)]
l5 = v1.loc[(v1['dorsal_ventral_ccf_coordinate'] <=1271) & (v1['dorsal_ventral_ccf_coordinate'] >=1117)]
l6 = v1.loc[(v1['dorsal_ventral_ccf_coordinate'] >=1271) & (v1['dorsal_ventral_ccf_coordinate'] <=1425)]

layers =[l1,l23,l4,l5,l6]
Layer_F_all = []
L_std_all = []
for t in range(len(tfs)):
    a = 475
    for l in layers:
        n = a+1-len(l)
        L = matrix_rep[n:a,2,t,0:rep,0:2000]
        a = n
        L_sumc = np.sum(L,axis=(1,2))
        L_single_F = L_sumc/(2+rep)
        #print(L_single_F.shape)
        L_std = np.std(L_single_F)
        L_F = np.mean(L_single_F)
        print(L_F.shape)
        Layer_F_all.append(L_F)
        L_std_all.append(L_std)
        
Layer_F_all = np.array(Layer_F_all)        
Layer_F_all = Layer_F_all.reshape(len(tfs),len(layers))

L_std_all = np.array(L_std_all)
L_std_all = L_std_all.reshape(len(tfs),len(layers))


color = ['moccasin','lightgreen','darkorange','peru','skyblue']
name = ['1','23','4','5','6']
for l in range(len(layers)):
    y = Layer_F_all[:,l]
    x = tfs #前面的定义
    yerr = L_std_all[:,l]
    #yerr = n1_std_all[t,:]
    plt.xlabel('Temporal frequence (Hz)')
    plt.ylabel('Response (spikes per second)')
    plt.suptitle('Response of diverse layers under differ tf drifting grating (oris:90 deg)')
    h1,= plt.plot(x,y,color[l],label='type2',marker = '*')
    plt.legend(handles=[h1], labels=['layer'+ name[l]],loc='best')
    #plt.errorbar(x,y,yerr,elinewidth=1,capsize=2)
    plt.savefig('E:/rotation_Jialab/Response of diverse layers under differ tf drifting grating oris 90')






#layer dependent oris FR
Layer_F_all = []
L_std_all = []
for o in range(len(oris)):
    a = 475
    for l in layers:
        n = a+1-len(l)
        L = matrix_rep[n:a,o,1,0:rep,0:2000]
        a = n
        L_sumc = np.sum(L,axis=(1,2))
        L_single_F = L_sumc/(2+rep)
        #print(L_single_F.shape)
        L_std = np.std(L_single_F)
        L_F = np.mean(L_single_F)
        print(L_F.shape)
        Layer_F_all.append(L_F)
        L_std_all.append(L_std)
        
Layer_F_all = np.array(Layer_F_all)        
Layer_F_all = Layer_F_all.reshape(len(oris),len(layers))

L_std_all = np.array(L_std_all)
L_std_all = L_std_all.reshape(len(oris),len(layers))

color = ['moccasin','lightgreen','darkorange','peru','skyblue']
name = ['1','23','4','5','6']
for l in range(len(layers)):
    y = Layer_F_all[:,l]
    x = oris #前面的定义
    yerr = L_std_all[:,l]
    #yerr = n1_std_all[t,:]
    plt.xlabel('Orientation (degrees)')
    plt.ylabel('Response (spikes per second)')
    plt.suptitle('Response of diverse layers under differ orientation drifting grating(tf:2Hz)')
    h1,= plt.plot(x,y,color[l],label='type2',marker = '*')
    plt.legend(handles=[h1], labels=['layer'+ name[l]],loc='best')
    #plt.errorbar(x,y,yerr,elinewidth=1,capsize=2)
    plt.savefig('E:/rotation_Jialab/Response of diverse layers under differ orientation drifting grating tf 2Hz')











#raster plot
import matplotlib.pyplot as plt
import numpy as np

# for t in range(len(tfs)):
#     for o in range(len(oris)):
for t in range(len(tfs)):
    for o in range(len(oris)):
        for n in range(10):
            n1 = matrix_rep[n,o,t,0:15,0:2000] 
            position = np.transpose(np.nonzero(n1)) # 这两个函数一起使用可以得到 ndarray 中非0值的坐标，第一列是reps, 第二列是所对应的时间。

            y = position[:,1]
            x = position[:,0]
            plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            plt.plot(y, x,'|', color='black')
            plt.xticks([0,2000])
            plt.title('Neuron {} Spike raster plot {} Hz {} deg'.format(n+1,tfs[t], oris[o]))
            plt.ylabel('trials')
            plt.xlabel('times')
            plt.savefig('E:/rotation_Jialab/Neuron {} Spike raster plot {} Hz {} deg.png'.format(n+1,str(tfs[t]), str(oris[o])))
            plt.clf()
        

       

#PSTH 
## differ width of time bins get differ shapes of PSTH

bin = [10,20,50,100]
binx_w_fs = []
# for t in range(len(tfs)):
    # for o in range(len(oris)):
for b in range(len(bin)):
    step = bin[0]
    n1 = np.sum(matrix_rep[0,0,0,0:15,0:2000],axis = 0)
    n1_bins = [n1[i:i+step] for i in range(0,len(n1),step)]
    n1_bins_averF = np.sum(n1_bins,axis = 1)/(bin[0]*1000)
    
    #plotting
    x = np.linspace(0,2000,200)
    N =len(x)
    
    # change x internal size
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = 12
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]
    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.title('Neuron 1 Peristimulus Time Histogram (1Hz, 0deg) '.format(step))
    plt.ylabel('Firing rate')
    plt.xlabel('times')
    plt.xticks(np.linspace(0,2000,200))
    plt.xticks(rotation=-15)    
    plt.plot(x,n1_bins_averF,color = 'black')
    plt.show()
    plt.savefig('E:/rotation_Jialab/Neuron 1 Peristimulus Time Histogram (1Hz, 0deg,{}time bins) '.format(step))
    plt.clf()



# Rsc
rsc_all = []
n1_averF=[]
n2_averF = []
for o in range(len(oris)):
    for b in range(len(bin)):
        n1 = np.sum(matrix_rep[0,o,0,0:15,50:bin[b]+50],axis = 1)
        n1_15_averF = n1/(bin[b]/1000)
        n1_averF.append(np.sum(n1_15_averF))

        n2 = np.sum(matrix_rep[1,o,0,0:15,50:bin[b]+50],axis = 1)
        n2_15_averF = n2/(bin[b]/1000)
        n2_averF.append(np.sum(n2_15_averF))

rsc_12 = np.corrcoef(n2_averF, n1_averF)
print(rsc_12) 
rsc_12 = round(rsc_12[0,1],7)
rsc_all.append(rsc_12)

      





   

    
    
    
    
    
    
    
        