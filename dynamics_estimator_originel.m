clear; clc; close all;
addpath("GaitPhaseEstimatorData/Daniel/")

%% Load Data
filename = "healthy_walking_treadmill_2kmph";
filename_initial_pos = "healthy_initial_posture_with_noise";
filename_force_noise = "healthy_treadmill_noise2";
filename_insole_points = "healthy_determine_points";

initial_pos = load("data_bags/" + filename_initial_pos + ".mat");
force_noise_calib = load("data_bags/" + filename_force_noise + ".mat");
insole_points =  load("data_bags/" + filename_insole_points + ".mat");
load("data_bags/" + filename + ".mat")

healthy_angle = load("healthy_ankle_angle.txt");

clear filename*
%% Initial Posture

initial_pos.t_foot = initial_pos.imu_data_foot.TimeStampGlob;
initial_pos.t_shank = initial_pos.imu_data_shank.TimeStampGlob;
initial_pos.q_foot = quaternion(initial_pos.imu_data_foot.QuatW, initial_pos.imu_data_foot.QuatX, initial_pos.imu_data_foot.QuatY, initial_pos.imu_data_foot.QuatZ);
initial_pos.q_foot_interp = quaternion(interp1(initial_pos.t_foot,[initial_pos.imu_data_foot.QuatW, initial_pos.imu_data_foot.QuatX, initial_pos.imu_data_foot.QuatY, initial_pos.imu_data_foot.QuatZ],initial_pos.t_shank));
initial_pos.q_shank = quaternion(initial_pos.imu_data_shank.QuatW, initial_pos.imu_data_shank.QuatX, initial_pos.imu_data_shank.QuatY, initial_pos.imu_data_shank.QuatZ);

for i=1:length(initial_pos.q_shank)
    initial_pos.ankle_angles(i) =  dist(initial_pos.q_foot_interp(i),initial_pos.q_shank(i));
end

initial_pos.ankle_angles(isnan(initial_pos.ankle_angles)) = 0;

initial_pos.ankle_angles_filt = smoothdata(initial_pos.ankle_angles,'movmedian','SmoothingFactor',0.1);

compensator = 0.16; %9 degrees
initial_angle = median(initial_pos.ankle_angles_filt(5:end-5)) + compensator;


%% Angle Estimator

t_foot = imu_data_foot.TimeStampGlob;
t_shank = imu_data_shank.TimeStampGlob;
q_foot = quaternion(imu_data_foot.QuatW, imu_data_foot.QuatX, imu_data_foot.QuatY, imu_data_foot.QuatZ);
q_foot_interp = quaternion(interp1(t_foot,[imu_data_foot.QuatW, imu_data_foot.QuatX, imu_data_foot.QuatY, imu_data_foot.QuatZ],t_shank));
q_shank = quaternion(imu_data_shank.QuatW, imu_data_shank.QuatX, imu_data_shank.QuatY, imu_data_shank.QuatZ);

for i=1:length(q_shank)
    ankle_angles(i) =  dist(q_foot_interp(i),q_shank(i)) - initial_angle;
end

ankle_angles(isnan(ankle_angles)) = 0;

ankle_angles_filt = smoothdata(ankle_angles,'movmedian','SmoothingFactor',0.1);

figure(1)
subplot(2,1,1)
plot(ankle_angles);
hold on; plot(ankle_angles_filt);
subplot(2,1,2)
plot(healthy_angle(:,1),healthy_angle(:,2))


%% vGRF Analyzer to determine Cutoff Frequency

% figure(2)
% plot(force_noise_calib.insole_data.Data)
% 
% for i=1:size(force_noise_calib.insole_data.Data,2)
%     data = double(force_noise_calib.insole_data.Data(:,i));
%     N = length(data);
%     Fs = 100; % Hz
% 
%     % FFT
%     Y = fft(data);
%     f = (0:N-1)*(Fs/N); 
% 
%     P2 = abs(Y/N);
%     P1 = P2(1:N/2+1);
%     P1(2:end-1) = 2*P1(2:end-1);
% 
%     figure(3);
%     plot(f(1:N/2+1), P1);
%     title('FFT of Signal');
%     xlabel('Frequency (Hz)');
%     ylabel('Magnitude');
%     grid on;
% 
%     % Select Cutoff Frequency
%     disp('Select the cutoff frequency from the plot');
%     [x, ~] = ginput(1);
%     cutoff_freq = x;
% 
%     % Low-Pass Filter
%     d = designfilt('lowpassfir', 'PassbandFrequency', cutoff_freq, ...
%                    'StopbandFrequency', cutoff_freq + 10, ...
%                    'PassbandRipple', 0.5, 'StopbandAttenuation', 65, ...
%                    'SampleRate', Fs);
% 
%     filtered_data = filter(d, data);
%     force_noise_calib.insole_data.DataFilt(:,i) = filtered_data;
% end

% figure;
% subplot(2, 1, 1);
% plot((0:N-1)/Fs, force_noise_calib.insole_data.Data);
% title('Original Signal');
% xlabel('Time (s)');
% ylabel('Amplitude');
% grid on;
% 
% subplot(2, 1, 2);
% plot((0:N-1)/Fs, force_noise_calib.insole_data.DataFilt);
% title('Filtered Signal');
% xlabel('Time (s)');
% ylabel('Amplitude');
% grid on;


%% vGRF Filter

cutoff_freq = 3;
Fs = 100;

d = designfilt('lowpassfir', 'PassbandFrequency', cutoff_freq, ...
                   'StopbandFrequency', cutoff_freq + 10, ...
                   'PassbandRipple', 0.5, 'StopbandAttenuation', 65, ...
                   'SampleRate', Fs);
for i=1:size(force_noise_calib.insole_data.Data,2)
    filtered_data = filter(d, double(force_noise_calib.insole_data.Data(:,i)));
    force_noise_calib.insole_data.DataFilt(:,i) = filtered_data;
    filtered_data = filter(d, double(insole_data.Data(:,i)));
    insole_data.DataFilt(:,i) = filtered_data;
end

figure(2);
subplot(2, 2, 1);
plot(force_noise_calib.insole_data.Data);
title('Original Signal Calibration');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(2, 2, 3);
plot(force_noise_calib.insole_data.DataFilt);
title('Filtered Signal Calibration');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(2, 2, 2);
plot(insole_data.Data);
title('Original Signal Walking');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(2, 2, 4);
plot(insole_data.DataFilt);
title('Filtered Signal Walking');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

%% Divide Insole Sensor in Regions
close;
figure(2);
subplot(3,1,1);
plot(insole_data.DataFilt(:,14)); hold on;
plot(insole_data.DataFilt(:,16)); 

plot(insole_data.DataFilt(:,12)); 
plot(insole_data.DataFilt(:,13)); 
plot(insole_data.DataFilt(:,15)); 
legend('14','16','12','13','15')

subplot(3,1,2);
plot(insole_data.DataFilt(:,6)); hold on;
plot(insole_data.DataFilt(:,10)); 
plot(insole_data.DataFilt(:,11)); 

plot(insole_data.DataFilt(:,7)); 
plot(insole_data.DataFilt(:,8)); 
plot(insole_data.DataFilt(:,9)); 
legend('6','10','11','7','8','9')


subplot(3,1,3);
plot(insole_data.DataFilt(:,3)); hold on;
plot(insole_data.DataFilt(:,4)); 
plot(insole_data.DataFilt(:,5)); 

plot(insole_data.DataFilt(:,1)); 
plot(insole_data.DataFilt(:,2));

legend('3','4','5','1','2')


heel = sum(insole_data.DataFilt(:,12:16),2);
heel(find(heel<300)) = 0;
mid = sum(insole_data.DataFilt(:,6:11),2);
mid(find(mid<300)) = 0;
tip = sum(insole_data.DataFilt(:,1:5),2);
tip(find(tip<300)) = 0;


%% vGRF Estimator

for i =1:length(heel)-200
    vGRF.data(i) = max(heel(i),max(mid(i),tip(i)));
end
vGRF.normalized = (vGRF.data - min(vGRF.data)) / (max(vGRF.data) - min(vGRF.data));
vGRF.time = insole_data.TimeStampGlob(1:end-200);

figure(3);
plot(vGRF.time, vGRF.normalized);
title('Normalized Vertical Ground Reaction Force (vGRF)');
xlabel('Time (s)');
ylabel('Normalized Force');
grid on;


%% Determine Gait Phases

