clear; clc; close all;

%% 1. Charger les données depuis un fichier MAT v7.3 (HDF5)
file_path = 'nia_walking2d5kmph.mat';

% Vérification des datasets disponibles
info = h5info(file_path);
disp({info.Datasets.Name}) % Affiche les noms des datasets

% Lire les données filtrées
dataset_path = '/insole_data_filtered/Data'; % Vérifiez avec h5disp si ce chemin est correct
data_test = h5read(file_path, dataset_path);

% Vérifier la taille de la matrice
disp(size(data_test));

% Vérifier si le nombre de colonnes est suffisant
num_cols = size(data_test,2);
if num_cols < 16
    error('Les données n''ont que %d colonnes, impossible d''accéder aux indices 12:16', num_cols);
end

% Extraction sécurisée des forces plantaires
heel_force = sum(data_test(:, 12:min(16, num_cols)), 2); % Talon
mid_force  = sum(data_test(:, 6:min(11, num_cols)), 2);  % Milieu
toe_force  = sum(data_test(:, 1:min(5, num_cols)), 2);   % Orteils

%% 2. Initialisation des variables pour l'analyse
cycle_starts = [];
gait_phases = cell(length(heel_force), 1);
phase_change_indices = [];
phase_labels = {};
in_cycle_TO = false;
in_cycle_FF = false;
total_force = zeros(size(heel_force));
true_heel_force = zeros(size(heel_force));
true_mid_force = zeros(size(heel_force));
true_toe_force = zeros(size(heel_force));
true_total_forces = zeros(size(heel_force));
percent_progression = zeros(size(heel_force));

%% 3. Calibrage et normalisation
known_force = 1;

for i = 1:length(heel_force)
    if (heel_force(i) > 333) || (mid_force(i) > 333) || (toe_force(i) > 333)
        total_force(i) = heel_force(i) + mid_force(i) + toe_force(i);
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

%% 4. Affichage des forces normalisées
figure;
plot(true_total_forces);
title('True Forces');
xlabel('Time');
ylabel('Force (N)');
legend('Normalized Force');
grid on;
hold on;

%% 5. Détection des phases de marche
for i = 1:length(heel_force)
    if true_heel_force(i) > 0.2 && (true_mid_force(i) < 0.1 || true_toe_force(i) < 0.1)
        phase = 'HS'; % Heel Strike
        if isempty(cycle_starts) || (i - cycle_starts(end)) > 100
            cycle_starts = [cycle_starts, i];
            in_cycle_TO = true;
            in_cycle_FF = true;
        end
    elseif true_heel_force(i) < 0.1 && true_mid_force(i) < 0.1 && true_toe_force(i) < 0.1
        phase = 'MSW'; % Mid Swing
    elseif true_heel_force(i) < 0.4 && true_mid_force(i) < 0.3 && true_toe_force(i) < 0.5
        if in_cycle_TO
            phase = 'TO'; % Toe Off
            in_cycle_TO = false;
        end
    elseif true_mid_force(i) > 0.3 && true_heel_force(i) < 0.3 && true_toe_force(i) > 0.25
        phase = 'HO'; % Heel Off
    elseif true_heel_force(i) > 0.25 && true_mid_force(i) > 0.25
        if in_cycle_FF
            phase = 'FF/MST'; % Flat Foot / Mid Stance
            in_cycle_FF = false;
        end
    end
    
    gait_phases{i} = phase;

    if i == 1 || ~strcmp(gait_phases{i}, gait_phases{i-1})
        phase_change_indices = [phase_change_indices, i];
        phase_labels = [phase_labels, phase];
    end
end

%% 6. Progression du cycle de marche
for c = 1:length(cycle_starts) - 1
    cycle_start = cycle_starts(c);
    cycle_end = cycle_starts(c + 1) - 1;
    cycle_length = cycle_end - cycle_start + 1;
    progression = linspace(0, 100, cycle_length); % 0% à 100%

    percent_progression(cycle_start:cycle_end) = progression;

    % Annotation des phases
    for i = cycle_start:cycle_end
        if ismember(i, phase_change_indices)
            phase_progression = round(progression(i - cycle_start + 1));
            max_force = max([true_heel_force(i), true_mid_force(i), true_toe_force(i)]);
            text(i, max_force + 0.02, sprintf('%s (%d%%)', gait_phases{i}, phase_progression), ...
                'FontSize', 8, 'HorizontalAlignment', 'center');
        end
    end
end

%% 7. Export des résultats pour usage externe
assignin('base', 'force_data', true_total_forces);
assignin('base', 'gait_phases', gait_phases);
assignin('base', 'phase_change_indices', phase_change_indices);
assignin('base', 'phase_labels', phase_labels);
assignin('base', 'gait_vector', percent_progression);
