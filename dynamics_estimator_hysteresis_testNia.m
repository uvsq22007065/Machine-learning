%% dynamics_estimator_hysteresis_testNia.m
clear; clc; close all;

%% 0. Paramètres et seuils
% Seuils pour la calibration et la détection des phases
FORCE_THRESHOLD       = 333;  % Seuil pour considérer la force significative
HEEL_STRIKE_THRESHOLD = 0.2;  % Seuil pour détecter le Heel Strike (HS)
MID_TOE_THRESHOLD     = 0.1;  % Seuil pour Mid Swing (MSW) et condition HS
TOE_OFF_THRESH_HEEL   = 0.4;  % Seuil pour Toe Off (TO) (force talon)
TOE_OFF_THRESH_MID    = 0.3;  % Seuil pour Toe Off (TO) (force milieu)
TOE_OFF_THRESH_TOE    = 0.5;  % Seuil pour Toe Off (TO) (force orteils)
HEEL_OFF_THRESH_HEEL   = 0.3;  % Seuil pour Heel Off (HO) (force talon)
HEEL_OFF_THRESH_TOE    = 0.25; % Seuil pour Heel Off (HO) (force orteils)
FF_THRESH_HEEL         = 0.25; % Seuil pour Flat Foot / Mid Stance (FF/MST) (force talon)
FF_THRESH_MID          = 0.25; % Seuil pour Flat Foot / Mid Stance (FF/MST) (force milieu)

% Force connue pour la calibration
known_force = 1;

%% 1. Correction des noms de champs et sauvegarde dans un nouveau fichier
originalFileName = 'nia_walking2d5kmph.mat';
fixedFileName    = 'nia_walking2d5kmph_fixed.mat';
correctMatFile(originalFileName, fixedFileName);

%% 2. Chargement des données corrigées
dataStruct = load(fixedFileName);
if ~isfield(dataStruct, 'insole_data_filtered')
    error('La variable "insole_data_filtered" n''est pas présente dans le fichier corrigé.');
end
insole_data = dataStruct.insole_data_filtered;

%% 3. Extraction des forces plantaires
[heel_force, mid_force, toe_force] = extractForces(insole_data);

%% 4. Calibration et normalisation des forces
[true_heel_force, true_mid_force, true_toe_force, true_total_forces] = calibrateForces(heel_force, mid_force, toe_force, FORCE_THRESHOLD, known_force);

%% 5. Affichage des forces normalisées
plotNormalizedForces(true_total_forces);

%% 6. Détection des phases de marche
[gait_phases, cycle_starts, phase_change_indices, phase_labels] = detectGaitPhases(...
    true_heel_force, true_mid_force, true_toe_force, ...
    HEEL_STRIKE_THRESHOLD, MID_TOE_THRESHOLD, ...
    TOE_OFF_THRESH_HEEL, TOE_OFF_THRESH_MID, TOE_OFF_THRESH_TOE, ...
    HEEL_OFF_THRESH_HEEL, HEEL_OFF_THRESH_TOE, ...
    FF_THRESH_HEEL, FF_THRESH_MID);

%% 7. Calcul de la progression du cycle de marche et annotation
percent_progression = computeGaitProgression(cycle_starts, phase_change_indices, gait_phases, true_heel_force, true_mid_force, true_toe_force);

%% 8. Export des résultats pour usage externe
assignin('base', 'force_data', true_total_forces);
assignin('base', 'gait_phases', gait_phases);
assignin('base', 'phase_change_indices', phase_change_indices);
assignin('base', 'phase_labels', phase_labels);
assignin('base', 'gait_vector', percent_progression);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonctions locales

function correctMatFile(originalFile, fixedFile)
    % Charge le fichier MAT original et renomme les champs non conformes.
    dataStruct = load(originalFile);
    fields = fieldnames(dataStruct);
    for i = 1:length(fields)
        currentField = fields{i};
        if ~isvarname(currentField)
            newField = matlab.lang.makeValidName(currentField);
            fprintf('Renommage de "%s" en "%s".\n', currentField, newField);
            dataStruct.(newField) = dataStruct.(currentField);
            dataStruct = rmfield(dataStruct, currentField);
        end
    end
    % Sauvegarde la structure corrigée dans un nouveau fichier MAT.
    save(fixedFile, '-struct', 'dataStruct');
    fprintf('Le fichier corrigé a été sauvegardé sous "%s".\n', fixedFile);
end

function [heelF, midF, toeF] = extractForces(insole_data)
    % Extraction des forces plantaires à partir de la matrice de données.
    % On suppose que la variable 'Data' contient les mesures.
    heelF = sum(insole_data.Data(:,12:16), 2); % Force au talon
    midF  = sum(insole_data.Data(:,6:11), 2);  % Force au milieu du pied
    toeF  = sum(insole_data.Data(:,1:5), 2);    % Force aux orteils
end

function [trueHeel, trueMid, trueToe, trueTotal] = calibrateForces(heel, mid, toe, threshold, known_force)
    % Calibrage et normalisation des forces.
    n = length(heel);
    trueHeel = zeros(n,1);
    trueMid  = zeros(n,1);
    trueToe  = zeros(n,1);
    trueTotal = zeros(n,1);
    for i = 1:n
        if (heel(i) > threshold) || (mid(i) > threshold) || (toe(i) > threshold)
            total = heel(i) + mid(i) + toe(i);
            trueHeel(i) = known_force * heel(i) / total;
            trueMid(i)  = known_force * mid(i)  / total;
            trueToe(i)  = known_force * toe(i)  / total;
            trueTotal(i) = max([trueHeel(i), trueMid(i), trueToe(i)]);
        else
            % Si aucune force significative, on fixe toutes les valeurs à 0.
            trueHeel(i) = 0;
            trueMid(i)  = 0;
            trueToe(i)  = 0;
            trueTotal(i) = 0;
        end
    end
end

function plotNormalizedForces(trueTotal)
    % Affichage des forces normalisées.
    figure;
    plot(trueTotal, 'LineWidth', 1.5);
    title('Forces normalisées');
    xlabel('Temps');
    ylabel('Force (N)');
    legend('Force Normalisée');
    grid on;
    hold on;
end

function [phases, cycleStarts, phaseChangeIdx, phaseLabels] = detectGaitPhases(...
        trueHeel, trueMid, trueToe, HS_thresh, midToe_thresh, ...
        TO_heel_thresh, TO_mid_thresh, TO_toe_thresh, ...
        HO_heel_thresh, HO_toe_thresh, FF_heel_thresh, FF_mid_thresh)
    % Détecte les phases de marche à partir des forces normalisées.
    n = length(trueHeel);
    phases = cell(n,1);
    cycleStarts = [];
    phaseChangeIdx = [];
    phaseLabels = {};
    
    % Variables de contrôle pour la détection des transitions
    inCycleTO = false;
    inCycleFF = false;
    
    for i = 1:n
        phase = 'Undefined';  % Valeur par défaut si aucune condition n'est remplie
        
        % Détection du Heel Strike (HS)
        if trueHeel(i) > HS_thresh && (trueMid(i) < midToe_thresh || trueToe(i) < midToe_thresh)
            phase = 'HS';
            if isempty(cycleStarts) || (i - cycleStarts(end)) > 100
                cycleStarts = [cycleStarts, i];
                inCycleTO = true;
                inCycleFF = true;
            end
            
        % Détection du Mid Swing (MSW)
        elseif trueHeel(i) < midToe_thresh && trueMid(i) < midToe_thresh && trueToe(i) < midToe_thresh
            phase = 'MSW';
            
        % Détection du Toe Off (TO)
        elseif trueHeel(i) < TO_heel_thresh && trueMid(i) < TO_mid_thresh && trueToe(i) < TO_toe_thresh
            if inCycleTO
                phase = 'TO';
                inCycleTO = false;
            end
            
        % Détection du Heel Off (HO)
        elseif trueMid(i) > HO_heel_thresh && trueHeel(i) < HO_heel_thresh && trueToe(i) > HO_toe_thresh
            phase = 'HO';
            
        % Détection du Flat Foot / Mid Stance (FF/MST)
        elseif trueHeel(i) > FF_heel_thresh && trueMid(i) > FF_mid_thresh
            if inCycleFF
                phase = 'FF/MST';
                inCycleFF = false;
            end
        end
        
        phases{i} = phase;
        % Enregistrement des indices où la phase change
        if i == 1 || ~strcmp(phases{i}, phases{i-1})
            phaseChangeIdx = [phaseChangeIdx, i];
            phaseLabels = [phaseLabels, phase];
        end
    end
end

function percentProg = computeGaitProgression(cycleStarts, phaseChangeIdx, phases, trueHeel, trueMid, trueToe)
    % Calcule la progression de chaque cycle de marche et annote les phases.
    n = length(trueHeel);
    percentProg = zeros(n,1);
    for c = 1:length(cycleStarts) - 1
        cycle_start = cycleStarts(c);
        cycle_end   = cycleStarts(c+1) - 1;
        cycle_length = cycle_end - cycle_start + 1;
        progression = linspace(0, 100, cycle_length);
        percentProg(cycle_start:cycle_end) = progression;
        
        % Annotation sur le graphique pour chaque changement de phase
        for i = cycle_start:cycle_end
            if ismember(i, phaseChangeIdx)
                phase_prog = round(progression(i - cycle_start + 1));
                max_force = max([trueHeel(i), trueMid(i), trueToe(i)]);
                text(i, max_force + 0.02, sprintf('%s (%d%%)', phases{i}, phase_prog), ...
                    'FontSize', 8, 'HorizontalAlignment', 'center');
            end
        end
    end
end
