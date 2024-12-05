% Lecture du rosbag
% Charger le rosbag
bag = rosbag('C:\Program Files\Data_bags\daniel_validation1d5kmph.bag'); % Modifier le chemin si nécessaire

% Extraction des données de vGRF_data
vGRF_topic = select(bag, 'Topic', '/vGRF'); % Nom du topic à vérifier
vGRF_msgs = readMessages(vGRF_topic, 'DataFormat', 'struct');

% Conversion des données de vGRF en vecteur numérique
vGRF_data = cellfun(@(msg) msg.Data, vGRF_msgs);

% Génération des timestamps simulés (100 Hz par défaut, ajuster si nécessaire)
frequency = 100; % Fréquence d'échantillonnage
time_stamps = (0:length(vGRF_data)-1) / frequency;

%% Calcul du gait_progress dans MATLAB
%% Initialisation des cycles et phases
cycle_starts = [];  % Stocker les indices où le cycle commence
gait_phases = cell(length(vGRF_data), 1);
phase_change_indices = [];
phase_labels = {};
in_stance_phase = false;

% Nouveau tableau pour accumuler percent_progression
all_percent_progressions = [];  % Table pour stocker la progression de chaque cycle

%% Détection des phases de marche : Stance et Swing
for i = 1:length(vGRF_data)
    if vGRF_data(i) > 0
        phase = 'Stance';
        if ~in_stance_phase
            in_stance_phase = true;
            cycle_starts = [cycle_starts, i];  % Début d'un nouveau cycle
        end
    else
        phase = 'Swing';
        in_stance_phase = false;
    end

    gait_phases{i} = phase;

    if i == 1 || ~strcmp(gait_phases{i}, gait_phases{i-1})
        phase_change_indices = [phase_change_indices, i];
        phase_labels = [phase_labels, phase];
    end
end

%% Calcul et accumulation de percent_progression pour chaque cycle
for c = 1:length(cycle_starts) - 1
    cycle_start = cycle_starts(c);
    cycle_end = cycle_starts(c+1) - 1;
    cycle_length = cycle_end - cycle_start + 1;
    progression = linspace(0, 100, cycle_length);  % Progression 0% à 100%
    all_percent_progressions = [all_percent_progressions, progression];  % Stocker la progression du cycle
end

%% Réutilisation des progressions accumulées pour affichage ou calcul
figure;
plot(vGRF_data, 'b'); hold on;
title('Phases de marche et progression');
xlabel('Temps (échantillons)');
ylabel('Force verticale (vGRF)');

for c = 1:length(cycle_starts) - 1
    progression = all_percent_progressions(cycle_starts(c):cycle_starts(c+1)-1);
    cycle_start = cycle_starts(c);
    for p = 10:10:100
        index_in_cycle = find(progression >= p, 1) + cycle_start - 1;
        if index_in_cycle <= length(vGRF_data)
            max_force_at_index = vGRF_data(index_in_cycle);
            text(index_in_cycle, max_force_at_index + 50, [num2str(p), '%'], ...
                'FontSize', 8, 'HorizontalAlignment', 'center');
        end
    end
end

%% Extraction du gait_progress depuis le rosbag
% Extraction du topic contenant gait_percentage_R (remplacer si nécessaire)
gait_topic = select(bag, 'Topic', '/gait_percentage_R');
gait_msgs = readMessages(gait_topic, 'DataFormat', 'struct');

% Conversion des données en vecteur numérique
gait_progress_rosbag = cellfun(@(msg) msg.Data, gait_msgs);

% Création d'un vecteur temporel pour le rosbag (normalisation)
time_rosbag = linspace(0, 1, length(gait_progress_rosbag));

% Création d'un vecteur temporel pour le gait_progress MATLAB (normalisation)
time_matlab = linspace(0, 1, length(all_percent_progressions));

% Vérifier et convertir les données en double
if iscell(gait_progress_rosbag)
    gait_progress_rosbag = cell2mat(gait_progress_rosbag);
end

if ~isa(gait_progress_rosbag, 'double')
    gait_progress_rosbag = double(gait_progress_rosbag);
end

% Interpolation après vérifications
gait_progress_rosbag_interp = interp1(time_rosbag, gait_progress_rosbag, time_matlab, 'linear', 'extrap');
all_percent_progressions_interp = interp1(time_matlab, all_percent_progressions, time_matlab, 'linear', 'extrap');

%% Calcul des métriques d'erreur
absolute_errors = abs(all_percent_progressions_interp - gait_progress_rosbag_interp);
squared_errors = (all_percent_progressions_interp - gait_progress_rosbag_interp).^2;

% Mean Absolute Error (MAE)
mae = mean(absolute_errors);

% Mean Squared Error (MSE)
mse = mean(squared_errors);

% Root Mean Squared Error (RMSE)
rmse = sqrt(mse);

% Affichage des résultats
disp(['Mean Absolute Error (MAE) : ', num2str(mae)]);
disp(['Mean Squared Error (MSE) : ', num2str(mse)]);
disp(['Root Mean Squared Error (RMSE) : ', num2str(rmse)]);

%% Comparaison graphique des deux vecteurs
figure;
plot(time_matlab, all_percent_progressions_interp, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Gait Progress MATLAB');
hold on;
plot(time_matlab, gait_progress_rosbag_interp, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Gait Progress Rosbag');
xlabel('Temps normalisé');
ylabel('Pourcentage de progression de la démarche');
title('Comparaison entre gait_progress MATLAB et Rosbag');
legend;
grid on;
hold off;

%% Histogramme des erreurs
figure;
histogram(absolute_errors, 'FaceColor', 'blue', 'EdgeColor', 'black');
xlabel('Erreurs absolues');
ylabel('Fréquence');
title('Distribution des erreurs absolues');
grid on;
