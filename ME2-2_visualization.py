#%%####################################### Import library  
import numpy as np
from scipy.io import wavfile
import mne
from mne_connectivity import read_connectivity
import matplotlib.pyplot as plt 

#%%####################################### Define functions
def ff(input_arr,target): # find the idx of the closest freqeuncy in freqs
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
    
def plot_SSEP(psds,freqs,color,title,level):
    if level == 'group':
        plot_err(psds,color,freqs)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.ylim([-0.2,1.6])
        plt.xlim([0.55,4])
        plt.title(title)
    elif level == 'individual':
        n_subjects = psds.shape[0]
        fig, axes = plt.subplots(5, 6, figsize=(18, 15), sharex=True, sharey=True)
        axes = axes.ravel()
        for subj in range(n_subjects):
            axes[subj].plot(freqs, psds[subj, :], color=color)
            axes[subj].set_title(f'S{subj+1}')
            axes[subj].axhline(y=0,
                               color='red',
                               linestyle='--')
            axes[subj].axvline(x=3.33,
                               color='red',
                               linestyle='--')
        # hide unused panels
        for ax in axes[n_subjects:]:
            ax.set_visible(False)
            plt.tight_layout()
        
def analyze_clusters(clu, src, label_names, label_v_ind, p=0.05):
    results = []
    sig_clu = np.where(clu[2] < p)[0]
    print("Found " + str(len(sig_clu)) + " significant clusters")
    for c in [clu[1][i] for i in sig_clu]:
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
folders = ['SSEP/','connectivity/'] 
analysis = ['fpsds','conn_plv']
which_data_type = ['_sensor_','_roi_redo4_','_morph_'] 

#%%####################################### Figure 1: Visualize the audio PSD  
fmin = 0.5
fmax = 4

plt.figure()
for f, c, l in zip(
    ['Duple300rr.wav', 'Triple300rr.wav'], # random control matched duple, matched triple, duple, triple
    ['#ff7f0e', '#1f77b4'],
    ['Duple', 'Triple']
):
    fs, audio = wavfile.read(root_path + 'stimuli/' + f)

    psds, freqs = mne.time_frequency.psd_array_welch(
        audio, fs,
        n_fft=len(audio),
        n_overlap=0,
        fmin=fmin,
        fmax=fmax
    )
    plt.plot(freqs, psds, color = c)
    plt.xlim([fmin, fmax])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(['Duple', 'Triple'])

#%%####################################### Figure 1: Visualize the sensor PSD 
n_folder = folders[0]
n_analysis = analysis[0]
data_type = which_data_type[0]
level = 'group'

for n_age in ages:
    print("Doing age " + n_age)
    duple = np.load(root_path + 'data/' + n_folder + n_age + '_group_03_rs_mag6pT' + data_type + n_analysis + '.npz') 
    triple = np.load(root_path + 'data/' + n_folder + n_age + '_group_04_rs_mag6pT' + data_type + n_analysis + '.npz') 
    
    freqs = duple[duple.files[1]]
    duple = duple[duple.files[0]]
    triple = triple[triple.files[0]]
    psds_duple = duple.mean(axis = 1)
    psds_triple = triple.mean(axis = 1)
        
    plt.figure()
    plot_SSEP(psds_duple,freqs,'#ff7f0e','',level)
    plot_SSEP(psds_triple,freqs,'#1f77b4','',level)
    plt.ylim([-0.23,1.4])
    plt.xlim([fmin, fmax])
    plt.legend(['Duple','', 'Triple',''])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
#%%####################################### Figure 2: Visualize ROI PSD and whole-brain MBR
## ROI
data_type = which_data_type[1]
n_analysis = analysis[0]
n_folder = folders[0]
level = 'group'
label_names = np.asarray(["Auditory", "SensoryMotor"])
nROI = np.arange(0,len(label_names),1)

for nn_age,n_age in enumerate(ages):
    print("Plotting age " + n_age)
    for n in nROI: 
        print("---------------------------------------------------Doing ROI: " + label_names[n])
        duple0 = np.load(root_path + 'data/' + n_folder + n_age + '_group_03_stc_rs_mne_mag6pT' + data_type + n_analysis + '.npz') 
        triple0 = np.load(root_path + 'data/' + n_folder + n_age + '_group_04_stc_rs_mne_mag6pT' + data_type + n_analysis + '.npz') 
        
        freqs = duple0[duple0.files[1]]  
        psds_duple = duple0[duple0.files[0]]
        psds_triple = triple0[triple0.files[0]]
        
        plt.figure()
        plot_SSEP(psds_duple[:,n,:],freqs,'#ff7f0e','',level)
        plot_SSEP(psds_triple[:,n,:],freqs,'#1f77b4',label_names[n],level)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

## whole-brain
data_type = which_data_type[2]
n_analysis = analysis[0]
n_folder = folders[0]
label_names = mne.get_volume_labels_from_aseg(root_path + 'subjects/aparc+aseg.mgz')

random = []
duple = []
triple = []

for n_age in ages:
    random0 = np.load(root_path + 'data/' + n_folder + n_age + '_group_02_stc_rs_mne_mag6pT_morph_' + n_analysis +'.npz')
    duple0 = np.load(root_path + 'data/' + n_folder + n_age + '_group_03_stc_rs_mne_mag6pT_morph_' + n_analysis +'.npz')
    triple0 = np.load(root_path + 'data/' + n_folder + n_age + '_group_04_stc_rs_mne_mag6pT_morph_' + n_analysis +'.npz')
    freqs = random0[random0.files[1]]         
    random.append(random0[random0.files[0]])
    duple.append(duple0[duple0.files[0]])
    triple.append(triple0[triple0.files[0]])
random = np.asarray(random)
duple = np.asarray(duple)
triple = np.asarray(triple)

MBR_duple = duple[1,:,:,ff(freqs,1.67)]/duple[1,:,:,ff(freqs,3.33)] - duple[0,:,:,ff(freqs,1.67)]/duple[0,:,:,ff(freqs,3.33)]
MBR_triple = triple[1,:,:,ff(freqs,1.11)]/triple[1,:,:,ff(freqs,3.33)] - triple[0,:,:,ff(freqs,1.11)]/triple[0,:,:,ff(freqs,3.33)]
T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.permutation_cluster_1samp_test(MBR_duple, seed = 0,verbose='CRITICAL')
results = analyze_clusters(clu, src, label_names, label_v_ind, p=0.05)
T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.permutation_cluster_1samp_test(MBR_triple, seed = 0,verbose='CRITICAL')
results = analyze_clusters(clu, src, label_names, label_v_ind, p=0.05)

p_threshold = 0.05

good_cluster_inds = np.where(clu[2] < p_threshold)[0]
good_clusters = [clu[1][idx] for idx in good_cluster_inds]
stc1 =  mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_mne_morph_mag6pT-vl.stc')
n_positions = len(stc1.data)
mask = np.zeros(n_positions, dtype=np.uint8)
positions = np.concatenate([x[0] for x in good_clusters])
mask[positions] = 1
stc1.data = np.vstack((mask,mask)).transpose()
stc1.plot(src=src,clim=dict(kind="value", lims=[0,1,2]))
  
#%%####################################### Figure 3: Visualize the ROI conn results 
n_folder = folders[1] 
n_analysis = analysis[1] # 1:'conn_plv'
data_type = which_data_type[1] # 1:_roi_redo4_

random_conn_all = []
duple_conn_all = []
triple_conn_all = []

nlines = 10
ROI1 = 1
ROI2 = 0
fmin = 5
fmax = 30
ymin = -0.1
ymax = 0.25

for n_age in ages:
    print("Doing connectivity " + n_age)
    random = read_connectivity(root_path + 'data/' + n_folder + n_age + '_group_02_stc_rs_mne_mag6pT' + data_type + n_analysis) 
    duple = read_connectivity(root_path + 'data/' + n_folder + n_age + '_group_03_stc_rs_mne_mag6pT' + data_type + n_analysis) 
    triple = read_connectivity(root_path + 'data/' + n_folder + n_age + '_group_04_stc_rs_mne_mag6pT' + data_type + n_analysis) 
    freqs = np.array(random.freqs)
    random_conn = random.get_data(output='dense')
    duple_conn = duple.get_data(output='dense')
    triple_conn = triple.get_data(output='dense')
    random_conn_all.append(random_conn)
    duple_conn_all.append(duple_conn)
    triple_conn_all.append(triple_conn)
    
print("-------------------Comparing 7 mo vs 11 mo in the duple-------------------")
conn1 = duple_conn_all[0]-random_conn_all[0] # 7mo
conn2 = duple_conn_all[1]-random_conn_all[1] # 11mo
XX = conn2 - conn1
    
plt.figure()
plot_err(conn1[:,ROI1,ROI2,:],'#D55E00',freqs)
plot_err(conn2[:,ROI1,ROI2,:],'#E69F00',freqs)
plt.xlim([fmin,fmax])
plt.ylim([ymin,ymax])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('7 mo vs. 11 mo Relatvie PLV between Auditory and Sensorimotor in the duple condition')
    
print("-------------------Comparing 7 mo vs 11 mo in the triple-------------------")
conn1 = triple_conn_all[0]-random_conn_all[0] # 7mo
conn2 = triple_conn_all[1]-random_conn_all[1] # 11mo
XX = conn2 - conn1
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(XX[:,ROI1,ROI2,:], seed = 0,verbose='CRITICAL') # test which frequency in Sensorimotor-Auditory is significant
good_cluster_inds = np.where(cluster_p_values < 0.1)[0]
print(cluster_p_values)
for i in np.arange(0,len(good_cluster_inds),1):
    print("The " + str(i+1) + "st significant cluster")
    print('Significant freqs: ' + str(freqs[clusters[good_cluster_inds[i]][0]]))

plt.figure()
plot_err(conn1[:,ROI1,ROI2,:],'#0072B2',freqs)
plot_err(conn2[:,ROI1,ROI2,:],'#56B4E9',freqs)
plt.vlines(x = freqs[clusters[good_cluster_inds[0]][0]],color='r',alpha = 0.2, ymin = ymin, ymax=ymax)
plt.vlines(x = freqs[clusters[good_cluster_inds[1]][0]],color='r',alpha = 0.2, ymin = ymin, ymax=ymax)

plt.xlim([fmin,fmax])
plt.ylim([ymin,ymax])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('7 mo vs. 11 mo Relatvie PLV between Auditory and Sensorimotor in the triple condition')

print("-------------------Comparing duple vs triple in the 7 mo-------------------")
conn1 = duple_conn_all[0]-random_conn_all[0] # 7mo
conn2 = triple_conn_all[0]-random_conn_all[0] # 7mo
XX = conn2 - conn1
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(XX[:,ROI1,ROI2,:], seed = 0,verbose='CRITICAL') # test which frequency in Sensorimotor-Auditory is significant
good_cluster_inds = np.where(cluster_p_values < 0.1)[0]
print(cluster_p_values)
for i in np.arange(0,len(good_cluster_inds),1):
    print("The " + str(i+1) + "st significant cluster")
    print('Significant freqs: ' + str(freqs[clusters[good_cluster_inds[i]][0]]))
    
plt.figure()
plot_err(conn1[:,ROI1,ROI2,:],'#D55E00',freqs)
plot_err(conn2[:,ROI1,ROI2,:],'#0072B2',freqs)
plt.vlines(x = freqs[clusters[good_cluster_inds[0]][0]],color='r',alpha = 0.2, ymin = ymin, ymax=ymax)
plt.xlim([fmin,fmax])
plt.ylim([ymin,ymax])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Duple vs. Triple Relatvie PLV between Auditory and Sensorimotor at 7 mo')
    
print("-------------------Comparing duple vs triple in the 11 mo-------------------")
conn1 = duple_conn_all[1]-random_conn_all[1] # 11mo
conn2 = triple_conn_all[1]-random_conn_all[1] # 11mo
XX = conn2 - conn1

plt.figure()
plot_err(conn1[:,ROI1,ROI2,:],'#E69F00',freqs)
plot_err(conn2[:,ROI1,ROI2,:],'#56B4E9',freqs)
plt.xlim([fmin,fmax])
plt.ylim([ymin,ymax])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Duple vs. Triple Relatvie PLV between Auditory and Sensorimotor at 11 mo')
