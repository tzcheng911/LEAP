#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:54:11 2025

Convert the SSEP, conn results to csv files for LMM analysis in R.

@author: tzcheng
"""
/media/tzcheng/storage2/CBS
#%%####################################### Import library  


#%%####################################### Define functions
def convert_Conn_to_csv(data_type,ROIs,n_analysis,n_folder,ROI1,ROI2):
    lm_np = []
    sub_col = [] 
    age_col = []
    cond_col = []
    ROI_col = []
    ages = ['7mo','11mo','br'] 
    conditions = ['_02','_03','_04']
    subj_path=['/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/' ,
               '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/11mo/',
               '/media/tzcheng/storage/BabyRhythm/']
         
    for n_age,age in enumerate(ages):
        print(age)
        for n_cond,cond in enumerate(conditions):
            print(cond)
            data0 = read_connectivity(root_path + n_folder + age + '_group' + cond + '_stc_rs_mne_mag6pT' + data_type + n_analysis) 
            freqs = data0.freqs
            data0_conn = data0.get_data(output='dense')
            print(np.shape(data0_conn))
            data = np.vstack((data0_conn[:,ROI1,ROI2,ff(freqs,1):ff(freqs,4)].mean(axis=-1),
                              data0_conn[:,ROI1,ROI2,ff(freqs,4):ff(freqs,8)].mean(axis=-1),
                              data0_conn[:,ROI1,ROI2,ff(freqs,8):ff(freqs,12)].mean(axis=-1),
                              data0_conn[:,ROI1,ROI2,ff(freqs,15):ff(freqs,30)].mean(axis=-1),
                              data0_conn[:,ROI1,ROI2,:].mean(axis=-1))).transpose() # delta, theta, alpha, beta, broadband, total 5 cols
            lm_np.append(data)
            if age == 'br':
                for file in os.listdir(subj_path[n_age]):
                    if file.startswith('br_'):
                        sub_col.append(file)
                        cond_col.append(cond)
                        age_col.append(age)
            else:
                for file in os.listdir(subj_path[n_age]):
                    if file.startswith('me2_'):
                        sub_col.append(file)
                        cond_col.append(cond)
                        age_col.append(age)
    lm_df = pd.DataFrame({'sub_id': sub_col,'age':age_col,'condition':cond_col,
                          'Delta conn': np.concatenate(lm_np)[:,0], 
                          'Theta conn': np.concatenate(lm_np)[:,1],
                          'Alpha conn': np.concatenate(lm_np)[:,2],
                          'Beta conn': np.concatenate(lm_np)[:,3],
                          'Broadband conn': np.concatenate(lm_np)[:,4]})
    lm_df.to_csv(root_path + n_folder + 'AM' + data_type + n_analysis + '.csv')
    
def convert_to_csv(data_type,ROIs,n_analysis,n_folder):
    lm_np = []
    sub_col = [] 
    age_col = []
    cond_col = []
    ages = ['7mo','11mo','br'] 
    conditions = ['_02','_03','_04']
    subj_path=['/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/' ,
               '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/11mo/',
               '/media/tzcheng/storage/BabyRhythm/'] # NEED TO BE THE ORDER OF where 7mo, 11mo and br data at
    if data_type == which_data_type[1] or data_type == which_data_type[2]:
        print('-----------------Extracting ROI data-----------------')
        ROI_col = []
    
        for n_age,age in enumerate(ages):
            print(age)
            for n_cond,cond in enumerate(conditions):
                print(cond)
                for nROI, ROI in enumerate(ROIs):
                    print(ROI)
                    data0 = np.load(root_path + n_folder + age + '_group' + cond + '_stc_rs_mne_mag6pT' + data_type + n_analysis +'.npz') 
                    data1 = data0[data0.files[0]]
                    print(np.shape(data1))
                    data2 = np.vstack((data1[:,nROI,[6,7]].mean(axis=1),data1[:,nROI,[12,13]].mean(axis=1),data1[:,nROI,[30,31]].mean(axis=1))).transpose()
                    lm_np.append(data2)
                    for file in os.listdir(subj_path[n_age]):
                        if file.startswith('7m') or file.startswith('11m') or file.startswith('br'):
                            sub_col.append(file)
                            cond_col.append(cond)
                            age_col.append(age)
                        else:
                            print('check the file')
        lm_df = pd.DataFrame({'sub_id': sub_col,'age':age_col,'condition':cond_col, 'ROI':ROI_col,'1.11Hz': np.concatenate(lm_np)[:,0], '1.67Hz': np.concatenate(lm_np)[:,1],'3.3Hz': np.concatenate(lm_np)[:,2]})
        lm_df.to_csv(root_path + n_folder + 'SSEP_roi.csv')
    elif data_type == which_data_type[0]:
        lm_np = []
        sub_col = [] 
        age_col = []
        cond_col = []
        print('-----------------Extracting sensor data-----------------')
        for n_age,age in enumerate(ages):
            print(age)
            for n_cond,cond in enumerate(conditions):
                print(cond)
                data0 = np.load(root_path + n_folder + age + '_group' + cond + '_rs_mag6pT' + data_type + n_analysis +'.npz') 
                data1 = data0[data0.files[0]].mean(axis=1)
                print(np.shape(data1))
                data2 = np.vstack((data1[:,[6,7]].mean(axis=1),data1[:,[12,13]].mean(axis=1),data1[:,[30,31]].mean(axis=1))).transpose()
                lm_np.append(data2)
                for file in os.listdir(subj_path[n_age]):
                    if file.startswith('7m') or file.startswith('11m') or file.startswith('br'):
                        sub_col.append(file)
                        cond_col.append(cond)
                        age_col.append(age)
                    else:
                        print('check the file')
        lm_df = pd.DataFrame({'sub_id': sub_col,'age':age_col,'condition':cond_col,'1.11Hz': np.concatenate(lm_np)[:,0], '1.67Hz': np.concatenate(lm_np)[:,1],'3.3Hz': np.concatenate(lm_np)[:,2]})
        lm_df.to_csv(root_path + n_folder + 'SSEP_sensor.csv')