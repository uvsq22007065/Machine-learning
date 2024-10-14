% Chemin vers le dossier contenant les fichiers .mat
mat_folder_path = 'data_bags/';  % Remplace par le chemin de ton dossier
csv_folder_path = 'csv_files/';    % Dossier où les fichiers .csv seront sauvegardés

% Créer le dossier pour les fichiers CSV s'il n'existe pas
if ~exist(csv_folder_path, 'dir')
    mkdir(csv_folder_path);
end

% Obtenir la liste de tous les fichiers .mat dans le dossier
mat_files = dir(fullfile(mat_folder_path, '*.mat'));

% Boucle sur chaque fichier .mat
for i = 1:length(mat_files)
    % Chemin complet vers le fichier .mat
    mat_file_path = fullfile(mat_folder_path, mat_files(i).name);
    
    % Charger les données du fichier .mat
    data = load(mat_file_path);
    
    % Obtenir les noms des variables dans le fichier .mat
    field_names = fieldnames(data);
    
    % Boucle sur chaque champ de la structure pour les traiter
    for j = 1:numel(field_names)
        % Obtenir les données du champ actuel
        current_data = data.(field_names{j});
        
        % Déterminer le nom du fichier CSV
        csv_file_name = [mat_files(i).name(1:end-4), '_', field_names{j}, '.csv'];
        csv_file_path = fullfile(csv_folder_path, csv_file_name);
        
        % Si les données sont un tableau numérique, on peut les enregistrer directement
        if isnumeric(current_data)
            csvwrite(csv_file_path, current_data);
        
        % Si les données sont une table, on peut les écrire directement en CSV
        elseif istable(current_data)
            writetable(current_data, csv_file_path);
            
        % Si les données sont une cellule, traiter les données
        elseif iscell(current_data)
            % Convertir la cellule en matrice, si possible
            try
                cell_data = cell2mat(current_data); % Peut échouer si la cellule contient des types variés
                csvwrite(csv_file_path, cell_data);
            catch
                disp(['Le champ ', field_names{j}, ' dans ', mat_files(i).name, ' ne peut pas être converti en matrice.']);
                cell2csv(csv_file_path, current_data); % Écrire les cellules en CSV
            end
            
        % Si les données sont une structure, essayer de les convertir en table
        elseif isstruct(current_data)
            try
                table_data = struct2table(current_data);
                writetable(table_data, csv_file_path);
            catch
                disp(['Le champ ', field_names{j}, ' dans ', mat_files(i).name, ' ne peut pas être converti en table.']);
            end
        else
            disp(['Le champ ', field_names{j}, ' dans ', mat_files(i).name, ' a un type non pris en charge : ', class(current_data)]);
        end
    end
end

disp('Conversion de tous les fichiers .mat en fichiers .csv terminée avec succès !');
