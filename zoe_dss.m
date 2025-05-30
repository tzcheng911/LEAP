clear 
close all

addpath(genpath('/home/tzcheng/Downloads/dss'))
root_path = '/media/tzcheng/storage2/CBS/mat/';
specific_path = 'MEG_f80450/ntrial_200/';
cd(strcat(root_path, specific_path,'dss_input'))

subjects = dir("*.mat");

for subj = 1:size(subjects)
    filename = subjects(subj).name;
    load(filename);
    meg = data(401:600,:,:); % trial * channels * time
    sr = 5000;
    times = linspace(-0.02, 0.2,1101);
    meg = permute(meg,[3, 2, 1]); % make it time * channels * trials
    c0=nt_cov(meg);
    c1=nt_cov(mean(meg,3)); % mean across trials as the bias
    [todss,pwr0,pwr1]=nt_dss0(c0,c1); % could select the number of PCs to keep (get rid of little ones)
    
    z=nt_mmat(meg,todss); % matrix multiplication to convert data to normalized DSS components 
    megclean2=nt_tsr(meg,z); % regress out to get clean data - project back to the sensor space
    megclean2 = permute(megclean2,[3, 2, 1]);
    save(strcat(root_path,specific_path,"dss_output/clean_pa_",filename),"megclean2");
    clear megclean2 meg filename
    strcat('Finish subject ',num2str(subj))
end

% avg = mean(megclean2,3);
% 
% figure;plot(times,squeeze(mean(meg,1)))
% hold on;plot(times,squeeze(mean(-megclean2*1e15,1)))
% xlim([-0.02 0.2])
% ylabel('Relative amplitude')
% xlabel('Time (s)')
% legend('EEG','"clean" EEG')
% 
% figure;imagesc(times,squeeze(meg));colormap jet; colorbar
% figure;imagesc(times,squeeze(megclean2));colormap jet; colorbar
