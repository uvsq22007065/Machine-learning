% Charger les données depuis le fichier CSV 
data = readtable('test_model.csv');

% Extraire les colonnes pertinentes
time = data.Time;               % Temps
force = data.Force;             % Force
gait_progress = data.Gait_Progress; % Progression de la démarche

% Ajuster le temps pour commencer à 0 secondes
time = time - time(1);

% Créer une figure
figure;

% Tracer la courbe de la force en fonction du temps
plot(time, force, 'b-', 'LineWidth', 2);
hold on;

% Ajouter les valeurs de Gait Progress toutes les 10 valeurs
for i = 1:10:length(force)
    text(time(i), force(i), sprintf('%.2f%%', gait_progress(i)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end


% Ajouter des légendes et des labels
xlabel('Temps (s)');
ylabel('Force');
title('Force en fonction du temps avec Gait Progress affiché');

% Afficher la figure
hold off;
