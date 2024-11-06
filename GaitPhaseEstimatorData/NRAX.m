% Définir les chemins des fichiers CSV pour les données d'entraînement et de test
train_file_path = 'C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/test_labels.csv';
test_file_path = 'C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/validation_labels.csv';

% Lire les fichiers CSV
train_data = readtable(train_file_path);
test_data = readtable(test_file_path);

force_data_train = train_data.Force;
force_Derivative_data_train = train_data.Force_Derivative;
gait_vector_train = train_data.Gait_Progress;
gait_phases_train = train_data.Phase;
ankle_angles_filt_train = train_data.Angle;
ankle_Derivative_angles_filt_train = train_data.Angle_Derivative;

force_data_test = test_data.Force;
force_Derivative_data_test = test_data.Force_Derivative;
gait_vector_test = test_data.Gait_Progress;
gait_phases_test = test_data.Phase;
ankle_angles_filt_test = test_data.Angle;
ankle_Derivative_angles_filt_test = test_data.Angle_Derivative;

% Préparer les caractéristiques d'entrée
X_train = [force_data_train, force_Derivative_data_train, ankle_angles_filt_train, ankle_Derivative_angles_filt_train];
X_test = [force_data_test, force_Derivative_data_test, ankle_angles_filt_test, ankle_Derivative_angles_filt_test];
y_train = gait_vector_train;
y_test = gait_vector_test;

% Normaliser les données
[X_train_scaled, mu, sigma] = zscore(X_train);
X_test_scaled = (X_test - mu) ./ sigma;

% Définir la longueur de la séquence
seq_length = 130;

% Créer des séquences
[X_seq_train, y_seq_train] = createSequences(X_train_scaled, y_train, seq_length);
[X_seq_test, y_seq_test] = createSequences(X_test_scaled, y_test, seq_length);

% Définir les couches du modèle avec MinLength
num_features = size(X_train, 2);  % Mise à jour pour prendre le nombre correct de caractéristiques
layers = [
    sequenceInputLayer(num_features, 'MinLength', seq_length)  % Utilisation de num_features
    convolution1dLayer(3, 128, 'Padding', 'same')
    reluLayer
    maxPooling1dLayer(2)
    convolution1dLayer(3, 64, 'Padding', 'same')
    reluLayer
    maxPooling1dLayer(2)
    lstmLayer(50, 'OutputMode', 'last')
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

% Options d'entraînement
options = trainingOptions('adam', ...
    'InitialLearnRate',0.0005, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',32, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress');

% Entraîner le modèle
net = trainNetwork(X_seq_train, y_seq_train, layers, options);

% Prédire sur les données de test
y_pred = predict(net, X_seq_test);

% Calculer les erreurs
mse = mean((y_pred - y_seq_test).^2);
mae = mean(abs(y_pred - y_seq_test));
fprintf('Mean Squared Error: %.2f\n', mse);
fprintf('Mean Absolute Error: %.2f\n', mae);

% Tracé de la progression réelle vs prédite
figure;
plot(y_seq_test, 'DisplayName', 'True gait progress');
hold on;
plot(y_pred, 'DisplayName', 'Prediction');
xlabel('Samples');
ylabel('Progression (%)');
title('Comparaison gait progress');
legend;
hold off;

% Scatter plot pour l'analyse des erreurs de prédiction
figure;
scatter(y_seq_test, y_pred, 'filled');
xlabel('True Gait Progress');
ylabel('Predicted');
title('True vs Predicted Gait Progress');
grid on;

% Fonction pour créer des séquences pour Conv1D et LSTM
function [X_seq, y_seq] = createSequences(data, labels, seq_length)
    % Ajuster le nombre de séquences pour éviter une erreur dimensionnelle
    num_sequences = size(data, 1) - seq_length + 1;
    
    % Initialisation de X_seq avec les bonnes dimensions
    X_seq = zeros(num_sequences, seq_length, size(data, 2));
    y_seq = zeros(num_sequences, 1);
    
    for i = 1:num_sequences
        X_seq(i, :, :) = data(i:i + seq_length - 1, :);  % Séquence de longueur seq_length
        y_seq(i) = labels(i + seq_length - 1);  % Dernier élément de la séquence pour la régression
    end
end
