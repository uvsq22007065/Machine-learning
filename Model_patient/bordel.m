% Charger les données depuis le fichier CSV
data = readtable('test_model.csv');

% Extraire les colonnes pertinentes
force = data.Force;              % Force
gait_progress = data.Gait_Progress; % Progression de la démarche
angle = data.Angle;              % Angle

% Créer une figure
figure;

% Tracer la courbe de la force en fonction de l'angle
plot(force, angle, 'b-', 'LineWidth', 2);
hold on;

% Ajouter des légendes et des labels
xlabel('Angle (degrés)');
ylabel('Force');
title('Force avec Gait Progress affiché sur la courbe');

% Afficher la figure
hold off;
