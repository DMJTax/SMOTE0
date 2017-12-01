
clear all;
clc;

run('datapath.m');

smote_datasets = dir(dpath);

dataset_counter = 1;
dataset_filenames = {};

fprintf('Datasets:\n');

fprintf('%2s: %6s | %6s | %6s | %5s | %s\n',...
    '#', 'N', 'N_min', 'N_maj', 'frac', 'dataset');
for j = 1:50
    fprintf('-');
end
fprintf('\n');

for j = 1:length(smote_datasets)
    
    temp_object = smote_datasets(j);
    
    if (temp_object.isdir == 1) % skip directories
        continue;
    end
    if (strcmp(temp_object.name,'info.html'))
        continue;
    end
    
    dataset_fname = temp_object.name;
    load([dpath,dataset_fname]);
    dim = size(a,2);
    
    [numbers,names] = classsizes(a);
    if (length(numbers) > 2)
        fprintf('dataset %s has more than 2 classes, skipping...\n',dataset_fname);
        continue;
    end
    
    [min_n,min_id] = min(numbers); % minority class size (n) and id
    [maj_n,maj_id] = max(numbers); % majority class size (n) and id
    % names of the classes
    min_name = names(min_id,:); 
    maj_name = names(maj_id,:);
    N = sum(numbers); % total # of objects
    balance = min_n / N; % balance of dataset
        
    fprintf('%2d: %6d | %6d | %6d | %1.3f | %s\n',dataset_counter,...
        N,min_n,maj_n,balance,dataset_fname)
    
    dataset_filenames{dataset_counter} = dataset_fname;
    dataset_counter = dataset_counter+1;
    
end

dataset_totry = input('What dataset do you want to experiment on? Type the id as it appears above: ');

dataset_totry_name = dataset_filenames{dataset_totry};

load([dpath,dataset_totry_name]);

oversampling_rate = input('Please input oversampling rate for minority class (1 = equal to majority class): ');

fprintf('Classifier number:\n1. LDA\n2. Parzen\n3. 1-NN\n4. SVM quadratic kernel\n');

classifer_number = input('Please type classifier number: ');

R = experiment(dataset_totry_name,classifer_number,oversampling_rate);

load([dpath,dataset_totry_name]);