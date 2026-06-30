#%%####################################### Import library  
import pickle
import numpy as np
import scipy.stats as stats
import mne
from mne_connectivity import read_connectivity
import matplotlib.pyplot as plt 

#%%####################################### Define functions
def ff(input_arr,target): 
    ## find the idx of the closest freqeuncy in freqs
    delta = 1000000000
    idx = -1
    for i, val in enumerate(input_arr):
        if abs(input_arr[i]-target) < delta:
            idx = i
            delta = abs(input_arr[i]-target)
    return idx

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)
    
def stats_SSEP(X,freqs,nonparametric):
    ## Compute parametric t-test and non-parametric 1D cluster test (across freqs) on X (psd1 - psd2)
    if nonparametric: 
        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, seed = 0,verbose='ERROR')
        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
        for i in np.arange(0,len(good_cluster_inds),1):
            print("The " + str(i+1) + "st significant cluster")
            print(clusters[good_cluster_inds[i]])
            print('Significant freqs: ' + str(freqs[clusters[good_cluster_inds[i]]]))
    else:   
        t,p = stats.ttest_1samp(X[:,ff(freqs,1.11)],0) # meter 1.11 Hz
        print('Testing freqs: ' + str(ff(freqs,1.11)))
        print('t statistics: ' + str(t))
        print('p-value: ' + str(p))
        t,p = stats.ttest_1samp(X[:,ff(freqs,1.67)],0) # meter 1.67 Hz
        print('Testing freqs: ' + str(ff(freqs,1.67)))
        print('t statistics: ' + str(t))
        print('p-value: ' + str(p))
        t,p = stats.ttest_1samp(X[:,ff(freqs,3.33)],0) # beat 3.3 Hz
        print('Testing freqs: ' + str(ff(freqs,3.33)))
        print('t statistics: ' + str(t))
        print('p-value: ' + str(p))

def wholebrain_spatio_temporal_cluster_test(X,n_meter,n_age,n_folder,p_threshold):
    ## Compute non-parametric 2D cluster test (across freqs and vertex) on X (psd1 - psd2) and save the cluster results
    print("Computing adjacency.")
    adjacency = mne.spatial_src_adjacency(src)
    
    ## set the cluster settings 
    df = np.shape(X)[0] - 1  # degrees of freedom for the test
    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)
    X = np.transpose(X,(0,2,1)) # subj, time, space            
    T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(
        X,
        seed=0,
        adjacency=adjacency,
        n_jobs=None,
        threshold=t_threshold,
        buffer_size=None,
        verbose=True,
    )
    filename = root_path + n_folder + n_age + '_fSSEP_wholebrain_cluster_' + n_meter +'.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(clu, f) # clu: clustering results of T_obs, clusters, cluster_p_values, H0

def stats_CONN(conn1,conn2,freqs,nlines,label_names,title,ROI1,ROI2,fmin,fmax):
    XX = conn1-conn2
    
    # non-parametric    
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(XX[:,ROI1,ROI2,:], seed = 0,verbose='ERROR') # test which frequency in Sensorimotor-Auditory is significant
    good_cluster_inds = np.where(cluster_p_values < 0.1)[0] ## set the cluster threshold to be p = 0.1 to observe marginal clusters too 
    print(cluster_p_values)
    for i in np.arange(0,len(good_cluster_inds),1):
        print("The " + str(i+1) + "st significant cluster")
        print('Significant freqs: ' + str(freqs[clusters[good_cluster_inds[i]][0]]))
    
    # parametric
    t,p = stats.ttest_1samp(XX[:,ROI1,ROI2,ff(freqs,fmin):ff(freqs,fmax)].mean(-1),0) 
    print('Significant freqs for uncorrected ttest (' + str(fmin) + '-' + str(fmax) + ' Hz): ' + 't-stats = ' + str(t))
    print('Significant freqs for uncorrected ttest (' + str(fmin) + '-' + str(fmax) + ' Hz): ' + 'p-value = ' + str(p))

def analyze_clusters(clu, src, label_names, label_v_ind, p=0.05):
    results = []
    for c in [clu[1][i] for i in np.where(clu[2] < p)[0]]:
        verts = c[-1]
        coords = np.round(src[0]['rr'][src[0]['vertno'][verts]] * 1000)

        roi_all = [label_names[j]
                   for v in verts
                   for j in range(len(label_names))
                   if v in label_v_ind[j][0]]

        res = {
            "cluster_stat": c[0],
            "min": coords.min(0),
            "max": coords.max(0),
            "mean": coords.mean(0),
            "ROIs_unique": list(set(roi_all)),
            "ROI_counts": {l: roi_all.count(l) for l in label_names}
        }

        print(
            c[0],
            "\nmin:", res["min"],
            "\nmax:", res["max"],
            "\nmean:", res["mean"],
            "\nROIs:", res["ROIs_unique"],
            "\n" + "\n".join(f"{res['ROI_counts'][l]} {l}" for l in label_names),
        )

        results.append(res)

    return results
#%%####################################### Set path
root_path = '/home/tzcheng/Desktop/ME2-2_upload_to_github/'

#%%####################################### set up the template brain
stc1 = mne.read_source_estimate(root_path + 'subjects/me2_101_7m_03_stc_mne_morph_mag6pT-vl.stc')
src = mne.read_source_spaces(root_path + 'subjects/fsaverage-vol-5-src.fif')
label_v_ind = np.load(root_path + 'subjects/ROI_lookup.npy', allow_pickle=True)
label_names = np.asarray(["Auditory", "SensoryMotor", "BG", "IFG"])
nROI = np.arange(0,len(label_names),1)

#%% Parameters
ages = ['7mo','11mo'] 
conditions = ['_02','_03','_04'] # random, duple, triple
folders = ['SSEP/','connectivity/'] 
analysis = ['fpsds','conn_plv']
which_data_type = ['_sensor_','_roi_redo4_','_morph_'] 

#%%####################################### Analysis on the sensor SSEP
n_folder = folders[0] # 0: SSEP
n_analysis = analysis[0] # 0: psds
data_type = which_data_type[0] # 1:_roi_ or 2:_roi_redo_

for n_age in ages:
    print("Doing age " + n_age)
    random = np.load(root_path + 'data/' + n_folder + n_age + '_group_02_rs_mag6pT' + data_type + n_analysis +'.npz') 
    duple = np.load(root_path + 'data/' + n_folder + n_age + '_group_03_rs_mag6pT' + data_type + n_analysis +'.npz') 
    triple = np.load(root_path +'data/' +  n_folder + n_age + '_group_04_rs_mag6pT' + data_type + n_analysis +'.npz') 
    freqs = random[random.files[1]]
    psds_random = random[random.files[0]].mean(axis = 1)
    psds_duple = duple[duple.files[0]].mean(axis = 1)
    psds_triple = triple[triple.files[0]].mean(axis = 1)
    print("-------------------Doing duple-------------------")
    stats_SSEP(psds_duple-psds_random,freqs,True)
    print("-------------------Doing triple-------------------")
    stats_SSEP(psds_triple-psds_random,freqs,True)

#%%####################################### Analysis on the ROI SSEP
n_folder = folders[0] # 0: SSEP
n_analysis = analysis[0] # 0: psds
data_type = which_data_type[1] # 1:_roi_ or 2:_roi_redo_

for n_age in ages:
    for n in nROI: 
        print("Doing ROI SSEP: " + label_names[n])
        random0 = np.load(root_path + 'data/' + n_folder + n_age + '_group_02_stc_rs_mne_mag6pT' + data_type + n_analysis +'.npz') 
        duple0 = np.load(root_path + 'data/' + n_folder + n_age + '_group_03_stc_rs_mne_mag6pT' + data_type + n_analysis + '.npz') 
        triple0 = np.load(root_path + 'data/' + n_folder + n_age + '_group_04_stc_rs_mne_mag6pT' + data_type + n_analysis + '.npz') 
        random = random0[random0.files[0]]
        duple = duple0[duple0.files[0]]
        triple = triple0[triple0.files[0]]
        freqs = random0[random0.files[1]]          
        print("-------------------Doing duple-------------------")
        stats_SSEP(duple[:,n,:]-random[:,n,:],freqs,True)
        print("-------------------Doing triple-------------------")
        stats_SSEP(triple[:,n,:]-random[:,n,:],freqs,True)
    
#%%####################################### Analysis on the wholebrain SSEP
n_folder = folders[0] # 0: SSEP
n_analysis = analysis[0] # 0: psds
data_type = which_data_type[2] # 2: morph data

p_threshold = 0.001 # set a cluster forming threshold based on a p-value for the cluster based permutation test p = 0.001 to replicate previous results
random = []
randomD = []
randomT = []
duple = []
triple = []

for n_age in ages:
    random0 = np.load(root_path + 'data/' +  n_folder + n_age + '_group_02_stc_rs_mne_mag6pT_morph_' + n_analysis +'.npz')
    duple0 = np.load(root_path + 'data/' +  n_folder + n_age + '_group_03_stc_rs_mne_mag6pT_morph_' + n_analysis +'.npz')
    triple0 = np.load(root_path + 'data/' +  n_folder + n_age + '_group_04_stc_rs_mne_mag6pT_morph_' + n_analysis +'.npz')
    freqs = random0[random0.files[1]]         
    random.append(random0[random0.files[0]])
    duple.append(duple0[duple0.files[0]])
    triple.append(triple0[triple0.files[0]])
random = np.asarray(random)
duple = np.asarray(duple)
triple = np.asarray(triple)

## Test whether meter and beat responses are significantly different between the two age groups
wholebrain_spatio_temporal_cluster_test(duple[1]-duple[0],'duple_11mo_7mo',n_age,n_folder,p_threshold)
wholebrain_spatio_temporal_cluster_test(triple[1]-triple[0],'triple_11mo_7mo',n_age,n_folder,p_threshold)

## Test whether MBRs are significantly different between the two age groups
label_names = mne.get_volume_labels_from_aseg(root_path + 'subjects/aparc+aseg.mgz')

MBR_duple = duple[1,:,:,ff(freqs,1.67)]/duple[1,:,:,ff(freqs,3.33)] - duple[0,:,:,ff(freqs,1.67)]/duple[0,:,:,ff(freqs,3.33)]
MBR_triple = triple[1,:,:,ff(freqs,1.11)]/triple[1,:,:,ff(freqs,3.33)] - triple[0,:,:,ff(freqs,1.11)]/triple[0,:,:,ff(freqs,3.33)]
T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.permutation_cluster_1samp_test(MBR_duple, seed = 0,verbose='CRITICAL')
results = analyze_clusters(clu, src, label_names, label_v_ind, p=0.05)
T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.permutation_cluster_1samp_test(MBR_triple, seed = 0,verbose='CRITICAL')
results = analyze_clusters(clu, src, label_names, label_v_ind, p=0.05)

#%%####################################### Analysis on the ROI conn
n_folder = folders[1] 
n_analysis = analysis[1] 
data_type = which_data_type[1] 

random_conn_all = []
randomD_conn_all = []
randomT_conn_all = []
duple_conn_all = []
triple_conn_all = []

nlines = 10
ROI1 = 1
ROI2 = 0
fmin = 5
fmax = 30

for n_age in ages:
    print("Doing connectivity at the age of " + n_age)
    random = read_connectivity(root_path + 'data/' +  n_folder + n_age + '_group_02_stc_rs_mne_mag6pT' + data_type + n_analysis) 
    duple = read_connectivity(root_path + 'data/' + n_folder + n_age + '_group_03_stc_rs_mne_mag6pT' + data_type + n_analysis) 
    triple = read_connectivity(root_path + 'data/' + n_folder + n_age + '_group_04_stc_rs_mne_mag6pT' + data_type + n_analysis) 
    freqs = np.array(random.freqs)
    random_conn = random.get_data(output='dense')
    duple_conn = duple.get_data(output='dense')
    triple_conn = triple.get_data(output='dense')
    random_conn_all.append(random_conn)
    duple_conn_all.append(duple_conn)
    triple_conn_all.append(triple_conn)
    ## Comparing connectivity (i.e. rhythmic vs. random)
    print("-------------------Doing duple-------------------")
    stats_CONN(duple_conn,random_conn,freqs,nlines,label_names,n_age + ' duple vs. random ' + n_analysis,ROI1,ROI2,fmin,fmax)
    print("-------------------Doing triple-------------------")
    stats_CONN(triple_conn,random_conn,freqs,nlines,label_names,n_age + ' triple vs. random ' + n_analysis,ROI1,ROI2,fmin,fmax)

## Comparing relatvie connectivity (i.e. rhythmic - random)
print("-------------------Comparing 7 mo vs 11 mo in the duple-------------------")
conn1 = duple_conn_all[0]-random_conn_all[0] # 7mo
conn2 = duple_conn_all[1]-random_conn_all[1] # 11mo
stats_CONN(conn1,conn2,freqs,nlines,label_names,'7 mo vs 11 mo',ROI1,ROI2,fmin,fmax)
print("-------------------Comparing 7 mo vs 11 mo in the triple-------------------")
conn1 = triple_conn_all[0]-random_conn_all[0] # 7mo
conn2 = triple_conn_all[1]-random_conn_all[1] # 11mo
stats_CONN(conn1,conn2,freqs,nlines,label_names,'7 mo vs 11 mo',ROI1,ROI2,fmin,fmax)
print("-------------------Comparing duple vs. triple in the 7mo-------------------")
conn1 = duple_conn_all[0]-random_conn_all[0] # 7mo duple
conn2 = triple_conn_all[0]-random_conn_all[0] # 7mo triple
stats_CONN(conn1,conn2,freqs,nlines,label_names,'Duple vs. Triple',ROI1,ROI2,fmin,fmax)
print("-------------------Comparing duple vs. triple in the 11 mo-------------------")
conn1 = duple_conn_all[1]-random_conn_all[1] # 11mo duple
conn2 = triple_conn_all[1]-random_conn_all[1] # 11mo triple
stats_CONN(conn1,conn2,freqs,nlines,label_names,'Duple vs. Triple',ROI1,ROI2,fmin,fmax)