%   R = EXPERIMENT(NAME,NR,FRAC)
%
% Do the noise-injection experiments on dataset NAME, using classifier
% NR:
% 1. LDA
% 2. Parzen
% 3. 1-NN
% 4. SVM quadratic kernel
%
% The minority class is upsampled to a fraction FRAC of the majority
% class. (I.e. when FRAC=1 both classes are balanced)

function R = experiment(dname,wnr,frac)

run('datapath.m'); % get the correct datapath
% if this file is missing, use datapath.m.example
% to create a file called 'datapath.m' for this machine

load([dpath,dname]);

%definition of classifiers and settings for classification:
% [DXD should be updated]
prwaitbar off;
prmemory inf;
reg = 1e-6;
switch wnr
case 1 
u = loglc2([],reg)*classc;
case 2 
u = scalem([],'variance')*parzenc*classc;
case 3 
u = scalem([],'variance')*knnc([],1);
case 4 
u = scalem([],'variance')*incsvc([],'p',2,10);
case 5 
u = randomforestc([],100,3)*classc;
end


%set other parameters and storage:
fname = sprintf('res_%s_classf%d_frac%.0f',dname,wnr,100*frac);
samplingnames = {'original';
   'balance priors';
   'ROS';
   'Parzen NI';
   'kNN NI';
};
nrfolds = 10; % number of folds for evaluation (outer folds)
kset = [1 5 10 15]; % number of neighbours to try (hyperparam)
nrkset = length(kset); % distinct possibilities for kset
nrintfolds = 5; % internal number of folds to use for opt. hyperparam.
optK = zeros(1,nrfolds); % the optimal K of each fold
err = NaN(4,2,nrfolds); 
% first dim = number of algorithms
% second    = AUC, MAP
% thurd     = folds

% start the loops:
I = nrfolds;
for i=1:nrfolds
	dd_message(3,'%d/%d ',i,nrfolds);
	[x,z,I] = dd_crossval(a,I); % x = trn, z = tst
	z = remclass(z); % why?

   % how many objects to generate?:
   n = size(x,1);            % trnsize
   m = sum(istarget(x));     % minority size
   N = ceil(frac*(n-m) - m); % samples to generate

   % train on orig. data
   w_tr = x*u;
   out = z*w_tr;
   err(1,1,i) = dd_auc(out);
   err(1,2,i) = dd_avprec(dd_prc(out));

   % adapt class priors
   xprior = setprior(x,[0.5 0.5]);
   w_tr = xprior*u;
   out = z*w_tr;
   err(2,1,i) = dd_auc(out);
   err(2,2,i) = dd_avprec(dd_prc(out));

   % train on random oversampling
   x_extra = gendat(target_class(x),N);
   w_tr = [x;x_extra]*u;
   out = z*w_tr;
   err(3,1,i) = dd_auc(out);
   err(3,2,i) = dd_avprec(dd_prc(out));

   % train on Parzen NI
   x_extra = gendatp(target_class(x),N);
   w_tr = [x;x_extra]*u;
   out = z*w_tr;
   err(4,1,i) = dd_auc(out);
   err(4,2,i) = dd_avprec(dd_prc(out));

   % train on kNN NI
   tmperr = zeros(nrkset,nrintfolds);
   Iint = nrintfolds;
   
   fprintf('Internal crossvalidation: ');
   for j=1:nrintfolds
      dd_message(4,'%d/%d ',j,nrintfolds);
      [xint, zint, Iint] = dd_crossval(x,Iint);
      for k=1:nrkset
         x_extra = gendatk(target_class(xint),N,kset(k));
         w_tr = [xint;x_extra]*u;
         out = zint*w_tr;
         tmperr(k,j) = dd_auc(out);
         % TODO: Internal cross validation should be on AUC if we evaluate
         % in terms of AUC, but it should be MAP if we evaluate MAP
      end
      
      % TODO: SMOTE
      % TODO: CBOS
      % TODO: ADOMS
      
      % TODO: PRIORS / REWEIGHTING OF CLASSES
      
      
   end
   % which performs best?
   [~,Kbest] = max(mean(tmperr,2));
   optK(i) = kset(Kbest);
   x_extra = gendatk(target_class(x),N,kset(Kbest));
   w_tr = [x;x_extra]*u;
   out = z*w_tr;
   
   err(4,1,i) = dd_auc(out);
   err(4,2,i) = dd_avprec(dd_prc(out));
   dd_message(4,'\n');
end
dd_message(3,'\n');

% and store everything nicely:
R = results(err,samplingnames,{'AUC' 'AP'},nrfolds);
R = setdimname(R,'upsampling','perf','run');
R = setname(R,fname);
save([rpath,fname],'R');

% And give some output to the command line:
S = average(100*R,3,'max1','dep');
show(S,'text','%4.1f');

optK
