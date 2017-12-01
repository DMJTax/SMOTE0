function s_pr = smote(x,N,k)
% s = SMOTE(x,N,k)
% Input:
% x, the dataset of minority samples
% N, the amount of artificial data to return
% k, the number of neighbours to use
% Output:
% s, the smoted data as a prdataset object

labels = getlab(x);
target_label = labels(1,:);

% first, randomly order the sample x
% this way, if N is not a multiple of the dataset size
% the result will not depend on the ordering of x
R = randperm(size(x,1))';
x = x(R,:);

T = size(x,1); % T = number of samples in x
D = pdist(x); % compute the T x T distance matrix
[~,nid] = sort(+D); % sort distances
neighbours = nid(2:end,:); % find neighbours, discard first row because
% that is the datapoint itself

% this matrix contains the id's of all nearest neighbours
% horizontal dim = the sample i
% vertical dim = the neighbours. the first row is the 1-NN,
% the second row is the 2-NN, etc. etc.

n_k = min(size(neighbours,1),k); % use n_k, since we may not have enough neighbours
if (n_k ~= k)
    warning('SMOTE requested %d neighbours, but there are only %d objects in the sample',k,T);
end
nn_matrix = neighbours(1:n_k,:); % the first n_k neighbour id's

s = NaN(N,size(x,2)); % create empty artificial data matrix

cursample = 1; % this iterator loops over all samples in x
while (N > 0) % N keeps track of how many artificial datapoints we still need to generate
    
    cur_neighbours = nn_matrix(:,cursample)'; % get the neighbours of the current sample
    
    temp_random_number = randi(n_k); % choose a random number between 1 and n_k
    chosen_neighbour = cur_neighbours(temp_random_number); % get neighbour id
    
    x_cn = x(chosen_neighbour,:); % randomly chosen neighbour
    x_cur = x(cursample,:);       % current data point
    
    alpha = rand(1); % interpolation factor
    
    x_new = alpha * x_cn + (1-alpha) * x_cur; % get artificial datapoint
    s(N,:) = x_new;
    
    cursample = cursample + 1; % go to next sample
    if (cursample > T) % start from start of dataset
        cursample = 1;
    end
    N = N - 1; 
    
end

s_pr = prdataset(s,repmat(target_label,size(s,1),1));

end

function smote_test()
    
    a = gendats(10);
    a = a(getlab(a)==2,:);
    figure;
    scatterd(a);
    
    N = 100;
    k = 200;
    s = smote(a,N,k);
    figure;
    scatterd(s);

end


