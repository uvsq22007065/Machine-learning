clear; clc; close all;
addpath("data_bags/")

%% Load Data
filename = "healthy_walking_treadmill_2kmph";
filename2 = "healthy_walking_treadmill_2kmph2";
filename3 = "healthy_walking_treadmill_2d5kmph";
filename_initial_pos = "healthy_initial_posture_with_noise";
filename_force_noise = "healthy_treadmill_noise2";
filename_insole_points = "healthy_determine_points";

initial_pos = load("data_bags/" + filename_initial_pos + ".mat");
force_noise_calib = load("data_bags/" + filename_force_noise + ".mat");
insole_points =  load("data_bags/" + filename_insole_points + ".mat");
load("data_bags/" + filename3 + ".mat")

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
% Synchronisation des données des IMUs (pied et tibia) et interpolation
t_foot = imu_data_foot.TimeStampGlob;
t_shank = imu_data_shank.TimeStampGlob;

% Création des quaternions pour le pied et le tibia
q_foot = quaternion(imu_data_foot.QuatW, imu_data_foot.QuatX, imu_data_foot.QuatY, imu_data_foot.QuatZ);
q_shank = quaternion(imu_data_shank.QuatW, imu_data_shank.QuatX, imu_data_shank.QuatY, imu_data_shank.QuatZ);

% Interpolation des quaternions du pied sur la même échelle temporelle que le tibia
q_foot_interp = quaternion(interp1(t_foot, [imu_data_foot.QuatW, imu_data_foot.QuatX, imu_data_foot.QuatY, imu_data_foot.QuatZ], t_shank));

% Initialisation de l'angle de référence (initial_angle)
initial_angle = 0; % ou une autre valeur calculée au besoin

% Préallocation du tableau des angles de la cheville
ankle_angles = zeros(1, length(q_shank));

% Calcul des angles entre le pied et le tibia
for i = 1:length(q_shank)
    ankle_angles(i) = dist(q_foot_interp(i), q_shank(i)) - initial_angle;
end

% Remplacement des valeurs NaN par 0
ankle_angles(isnan(ankle_angles)) = 0;

% Filtrage des angles avec un filtre de médiane mobile
ankle_angles_filt = smoothdata(ankle_angles, 'movmedian', 'SmoothingFactor', 0.1);

% Affichage ou exportation des résultats
assignin('base','ankle_angles_filt', ankle_angles_filt);

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

for i = 1:length(heel)-200
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
%% Calibrage
% Initialisation des variables
heel_force = sum(insole_data.DataFilt(:,12:16), 2);
mid_force = sum(insole_data.DataFilt(:,6:11), 2);
toe_force = sum(insole_data.DataFilt(:,1:5), 2);
cycle_starts = []; % Stocker les indices où le cycle commence
gait_phases = cell(length(heel_force), 1);
phase_change_indices = []; % Indices de changements de phase
phase_labels = {}; % Labels des phases
in_cycle_TO = false;  % Indicateur de cycle
in_cycle_FF = false;
total_force = zeros(size(heel_force)); % Assurez-vous que total_force a la bonne taille
true_heel_force = zeros(size(heel_force));
true_mid_force = zeros(size(heel_force));
true_toe_force = zeros(size(heel_force));
true_total_forces = zeros(size(heel_force));
percent_progression = [];

% Calibrage
known_force = 1;

for i = 1:length(heel_force)  % Itération sur chaque index
    if (heel_force(i) > 333) || (mid_force(i) > 333) || (toe_force(i) > 333)
        total_force(i) = heel_force(i) + mid_force(i) + toe_force(i);  % Corriger ici
        true_heel_force(i) = known_force * heel_force(i) / total_force(i);
        true_mid_force(i) = known_force * mid_force(i) / total_force(i);
        true_toe_force(i) = known_force * toe_force(i) / total_force(i);
        true_total_forces(i) = max([true_heel_force(i), true_mid_force(i), true_toe_force(i)]);
    else
        true_heel_force(i) = 0;
        true_toe_force(i) = 0;
        true_mid_force(i) = 0;
        true_total_forces(i) = 0;
    end
end

% Maintenant, plottez true forces
figure; % Crée une nouvelle figure pour éviter les superpositions
% plot(true_heel_force, 'b'); hold on;
% plot(true_mid_force, 'g'); 
% plot(true_toe_force, 'r'); 
plot(true_total_forces)
title('True Forces');
xlabel('Time');
ylabel('Force (N)');
legend('Normalized Force');
% legend('Force du talon', 'Force du médio-pied', 'Force des orteils', 'Force maximale');
grid on;
hold on;

% Détection des phases de marche en fonction des conditions sur les forces
for i = 1:length(heel_force)
    if true_heel_force(i) > 0.2 && (true_mid_force(i) < 0.1 || true_toe_force(i) < 0.1)
        phase = 'HS';
        
        % Enregistrer le début d'un nouveau cycle lors d'un Heel Strike
        if isempty(cycle_starts) || (i - cycle_starts(end)) > 100  % Assurer qu'on ne détecte pas plusieurs fois le même cycle
            cycle_starts = [cycle_starts, i];
            in_cycle_TO = true;  % Indiquer que nous sommes maintenant dans un cycle
            in_cycle_FF = true;
        end
    elseif true_heel_force(i) < 0.1 && true_mid_force(i) < 0.1 && true_toe_force(i) < 0.1
        phase = 'MSW';
        
    elseif true_heel_force(i) < 0.4 && true_mid_force(i) < 0.3 && true_toe_force(i) < 0.5
        if in_cycle_TO  % Vérifiez si nous sommes toujours dans le cycle
            phase = 'TO';
            in_cycle_TO = false;  % Sortir du cycle après détection de TO
        end
        
    elseif true_mid_force(i) > 0.3 && true_heel_force(i) < 0.3 && true_toe_force(i) > 0.25
        phase = 'HO';
             
    elseif true_heel_force(i) > 0.25 && true_mid_force(i) > 0.25
        if in_cycle_FF  % Vérifiez si nous sommes toujours dans le cycle
            phase = 'FF/MST';
            in_cycle_FF = false;  % Sortir du cycle après détection de TO
        end
    end
    
    gait_phases{i} = phase;

    % Détecter les changements de phase
    if i == 1 || ~strcmp(gait_phases{i}, gait_phases{i-1})
        phase_change_indices = [phase_change_indices, i]; % Enregistrer les indices de changement de phase
        phase_labels = [phase_labels, phase]; % Enregistrer les labels des phases
    end
end

% Fusion des calculs et annotations pour chaque cycle détecté
for c = 1:length(cycle_starts) - 1
    cycle_start = cycle_starts(c);
    cycle_end = cycle_starts(c + 1) - 1;
    cycle_length = cycle_end - cycle_start + 1;
    progression = linspace(0, 100, cycle_length); % Progression de 0% à 100% pour chaque cycle

    % Affectation des valeurs de progression au tableau `percent_progression`
    percent_progression(cycle_start:cycle_end) = progression;

    % Marquer les changements de phase pour ce cycle
    for i = cycle_start:cycle_end
        if ismember(i, phase_change_indices)
            % Calcul du pourcentage de progression pour l'index du changement de phase
            phase_progression = round(progression(i - cycle_start + 1));  % Pourcentage correspondant à l'index
            max_force = max([true_heel_force(i), true_mid_force(i), true_toe_force(i)]);

            % Annoter avec phase et pourcentage
            text(i, max_force + 0.02, sprintf('%s (%d%%)', gait_phases{i}, phase_progression), ...
                'FontSize', 8, 'HorizontalAlignment', 'center');
        end
    end

    % Marquer les dizaines de pourcent (10%, 20%, ...) directement sur la courbe pour chaque cycle
    for p = 10:10:100
        index_in_cycle = find(progression >= p, 1) + cycle_start - 1;
        max_force_at_index = max([true_heel_force(index_in_cycle), true_mid_force(index_in_cycle), true_toe_force(index_in_cycle)]);
        
        % Ajouter les pourcentages au-dessus des courbes
        text(index_in_cycle, max_force_at_index + 0.01, [num2str(p), '%'], 'FontSize', 8, 'HorizontalAlignment', 'center');
    end
end

clean_percent_progression = percent_progression(~isnan(percent_progression));

% Output the force data and gait phases to the Matlab workspace for Python access
assignin('base', 'force_data', true_total_forces);  % Output all force data
assignin('base', 'gait_phases', gait_phases);       % Output gait phases based on force conditions
assignin('base', 'phase_change_indices', phase_change_indices);  % Store phase change indices for reference
assignin('base', 'phase_labels', phase_labels);  % Store labels for phases
assignin('base', 'gait_vector', clean_percent_progression);
assignin('base','ankle_angles_shank_filt', filtered_shank);
assignin('base','ankle_angles_foot_filt', filtered_foot);

%% Dérivées
% Calcul des dérivées
force_derivative = diff(true_total_forces);  % Dérivée des forces
angle_derivative = diff(ankle_angles_filt);   % Dérivée des angles

% Pour afficher, nous devons ajuster la longueur des dérivées
time_vector_force = (0:length(true_total_forces)-2)';  % Vecteur temps pour les forces
time_vector_angle = (0:length(ankle_angles_filt)-2)';   % Vecteur temps pour les angles

% Création de la figure
figure;

% Tracé des dérivées des forces
subplot(2, 1, 1);  % Création d'un sous-graphique
plot(time_vector_force, force_derivative, 'b');  % Dérivée des forces en bleu
title('Dérivée des Forces');
xlabel('Temps');
ylabel('Dérivée de la Force (N/s)');
grid on;

% Tracé des dérivées des angles
subplot(2, 1, 2);  % Création d'un autre sous-graphique
plot(time_vector_angle, angle_derivative, 'r');  % Dérivée des angles en rouge
title('Dérivée des Angles');
xlabel('Temps');
ylabel('Dérivée de Angle (rad/s)');
grid on;