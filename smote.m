function s = smote(x,N,k)
% s = SMOTE(x,N,k)
% Input:
% x, the dataset of minority samples
% N, the amount of SMOTE.
% This is assumed to be divisible by 100 if N > 100.
%
% k, the number of neighbours to use
% Output:
% s, the smoted data

% first, randomly order the sample x (incase N < 100)
R = randperm(size(x,1))';
x = x(R,:);

T = size(x,1); % T = number of samples

D = pdist(x); % compute the T x T distance matrix

[distances,nid] = sort(+D); % sort distances

neighbours = nid(2:end,:); % find neighbours

% this matrix contains the id's of all nearest neighbours
% horizontal dim = the sample i
% vertical dim = the neighbours. the first row is the 1-NN,
% the second row is the 2-NN, etc. etc.

nn_matrix = neighbours(1:k,:); % the first k neighbour id's

N = round(N/100);

end




