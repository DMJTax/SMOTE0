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

% dpath = '/data/smote0/';
% dpath = '/tudelft.net/staff-groups/ewi/insy/PRLab/data/smote0/';
load('datapath.m');
load([dpath,dname]);

%settings for classification:
prwaitbar off;
nrfolds = 10;
reg = 1e-6;
switch wnr
case 1 
u = ldc([],reg,reg)*classc;
case 2 
u = scalem([],'variance')*parzenc*classc;
case 3 
u = scalem([],'variance')*knnc([],1);
case 4 
u = scalem([],'variance')*incsvc([],'p',2,10);
end


%set other parameters and storage:
fname = sprintf('res_%s_classf%d_frac%.0f',dname,wnr,100*frac);
samplingnames = {'org';
'ROS';
'Parzen NI';
'kNN NI';
};
perf = repmat(NaN,[3 2 nrfolds]);

% start the loops:
I = nrfolds;
for i=1:nrfolds
	dd_message(3,'%d/%d ',i,nrfolds);
	[x,z,I] = dd_crossval(a,I);
	z = remclass(z);

   % how many objects to generate?:
   n = size(x,1);
   m = sum(istarget(x));
   N = ceil(frac*(n-m) - m);

   % train on orig. data
   w_tr = x*u;
   out = z*w_tr;
   err(1,1,i) = dd_auc(out);
   err(1,2,i) = dd_avprec(dd_prc(out));

   % train on random oversampling
   x_extra = gendat(target_class(x),N);
   w_tr = [x;x_extra]*u;
   out = z*w_tr;
   err(2,1,i) = dd_auc(out);
   err(2,2,i) = dd_avprec(dd_prc(out));

   % train on Parzen NI
   x_extra = gendatp(target_class(x),N);
   w_tr = [x;x_extra]*u;
   out = z*w_tr;
   err(3,1,i) = dd_auc(out);
   err(3,2,i) = dd_avprec(dd_prc(out));

   % train on kNN NI
   x_extra = gendatk(target_class(x),N);
   w_tr = [x;x_extra]*u;
   out = z*w_tr;
   err(4,1,i) = dd_auc(out);
   err(4,2,i) = dd_avprec(dd_prc(out));

end
dd_message(3,'\n');

% and store everything nicely:
R = results(err,samplingnames,{'AUC' 'AP'},nrfolds);
R = setdimname(R,'upsampling','perf','run');
R = setname(R,fname);
save(fname,'R');

% And give some output to the command line:
fprintf('\n%s\n\n',repmat('=',1,50));
a
S = average(100*R,3,'max1','dep');
show(S,'text','%4.1f');

