clear 
close all
data = load('cbs_A103_epoch.mat');
meg(1,:,:) = data.data(1:200,:);
sr = 5000;
times = linspace(-0.02, 0.2,1101);
meg = permute(meg,[2, 1, 3]);
c0=nt_cov(meg);
c1=nt_cov(mean(meg,3));
[todss,pwr0,pwr1]=nt_dss0(c0,c1);

z=nt_mmat(meg,todss);
megclean2=nt_tsr(meg,z); % regress out to get clean data

avg = mean(megclean2,3);

figure;plot(times,squeeze(mean(meg,1)))
hold on;plot(times,squeeze(mean(-megclean2*1e15,1)))
xlim([-0.02 0.2])
ylabel('Relative amplitude')
xlabel('Time (s)')
legend('EEG','"clean" EEG')

figure;imagesc(times,squeeze(meg));colormap jet; colorbar
figure;imagesc(times,squeeze(megclean2));colormap jet; colorbar
