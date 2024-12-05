%% Charger les rosbags
bag1 = rosbag('C:\Program Files\Data_bags\daniel_validation1d5kmph.bag'); % Modifier le chemin si nécessaire
bag2 = rosbag('C:\Program Files\Data_bags\walking_exoskeleton_force_control.bag'); % Modifier avec le chemin du 2e bag
bag3 = rosbag('C:\Program Files\Data_bags\daniel_validation1kmph.bag'); % Modifier le chemin si nécessaire

%% Extraction des données du topic /ankle
ankle_topic1 = select(bag1, 'Topic', '/insole_data');
ankle_topic2 = select(bag2, 'Topic', '/insole_data');
ankle_topic3 = select(bag3, 'Topic', '/insole_data');

ankle_msgs1 = readMessages(ankle_topic1, 'DataFormat', 'struct');
ankle_msgs2 = readMessages(ankle_topic2, 'DataFormat', 'struct');
ankle_msgs3 = readMessages(ankle_topic3, 'DataFormat', 'struct');

%% Conversion des données en vecteurs numériques
ankle_data1 = cellfun(@(msg) msg.Data, ankle_msgs1, 'UniformOutput', false); % Assurez-vous que msg.Data correspond bien
ankle_data2 = cellfun(@(msg) msg.Data, ankle_msgs2, 'UniformOutput', false); % Idem pour le second rosbag
ankle_data3 = cellfun(@(msg) msg.Data, ankle_msgs3, 'UniformOutput', false); % Idem pour le second rosbag

% Si les données sont des tableaux ou vecteurs, les convertir en format numérique simple
ankle_data1 = cellfun(@(x) x(:), ankle_data1, 'UniformOutput', false);
ankle_data2 = cellfun(@(x) x(:), ankle_data2, 'UniformOutput', false);
ankle_data3 = cellfun(@(x) x(:), ankle_data3, 'UniformOutput', false);

% Fusionner les données en un seul vecteur
ankle_data1 = cell2mat(ankle_data1);
ankle_data2 = cell2mat(ankle_data2);
ankle_data3 = cell2mat(ankle_data3);

moyenne1 = mean(ankle_data1)
moyenne2 = mean(ankle_data2)
moyenne3 = mean(ankle_data3)

%% Synchronisation des données (en supposant une fréquence de 100 Hz)
frequency = 100; % Fréquence d'échantillonnage (ajustez si nécessaire)
time_stamps1 = (0:length(ankle_data1)-1) / frequency;
time_stamps2 = (0:length(ankle_data2)-1) / frequency;
time_stamps3 = (0:length(ankle_data3)-1) / frequency;

% Rééchantillonnage si les tailles sont différentes
min_length = min(length(ankle_data1), length(ankle_data2));
ankle_data1 = ankle_data1(1:min_length) - moyenne1;
ankle_data2 = ankle_data2(1:min_length) - moyenne2;
ankle_data3 = ankle_data3(1:min_length) - moyenne3;
time_stamps = (0:min_length-1) / frequency;

%% Calcul de la différence
ankle_difference = ankle_data1 - ankle_data2;
ankle_difference2 = ankle_data3 - ankle_data2;

%% Tracer les résultats
figure;

% Plot des deux séries de données
subplot(2, 1, 1);
% plot(time_stamps, ankle_data1, 'b', 'DisplayName', 'Ankle without exo 1.5 kmph');
% hold on;
plot(time_stamps, ankle_data2, 'r', 'DisplayName', 'Ankle with exo');
hold on;
plot(time_stamps, ankle_data3, 'k', 'DisplayName', 'Ankle without exo 1 kmph');
xlabel('Time (s)');
ylabel('Ankle Data');
title('Comparaison des données Ankle');
legend;
grid on;

% Plot de la différence
subplot(2, 1, 2);
% plot(time_stamps, ankle_difference, 'k', 'DisplayName', 'Différence 1.5 kmph minus exo');
% hold on;
plot(time_stamps, ankle_difference2, 'b', 'DisplayName', 'Différence 1 kmph minus exo');
xlabel('Time (s)');
ylabel('Différence');
title('Différence entre les deux jeux de données');
legend;
grid on;
