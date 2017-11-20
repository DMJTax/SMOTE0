% Get a one-class dataset where the target class is small
a = oc_set(gendatb([10,50]),1);
a = oc_set(gendatd([10,50]),1);

% settings for the noise injection
N = 100;   % nr. of objects to add

%settings for classification:
prwaitbar off;
nrfolds = 10;
reg = 1e-6;
switch wnr
case 1 
 u = ldc([],reg,reg)*classc;
	  scalem([],'variance')*parzenc*classc;
	  scalem([],'variance')*knnc([],1);
	  scalem([],'variance')*incsvc([],'p',2,10)};
     };
wnames = getwnames(w);


%set other parameters and storage:
nrw = length(w);
perf = repmat(NaN,[nrw 3 nrfolds]);

% start the loops:
I = nrfolds;
for i=1:nrfolds
	dd_message(3,'%d/%d ',i,nrfolds);
	[x,z,I] = dd_crossval(a,I);
	z = remclass(z);

	for j=1:nrw
      j
      % train on orig. data
		w_tr = x*w{j};
		out = z*w_tr;
		err(j,1,i) = dd_auc(out);
      % train on Parzen NI
      x_extra = gendatp(target_class(x),N);
		w_tr = [x;x_extra]*w{j};
		out = z*w_tr;
		err(j,2,i) = dd_auc(out);
      % train on Parzen NI
      x_extra = gendatk(target_class(x),N);
		w_tr = [x;x_extra]*w{j};
		out = z*w_tr;
		err(j,3,i) = dd_auc(out);
	end
end
dd_message(3,'\n');

% and store everything nicely:
if isempty(wnames) wnames = getwnames(w); end
R = results(err,wnames,{'org' 'ParzenNI' 'kNN NI'},nrfolds);
R = setdimname(R,'classifier','upsampling','run');
R = setname(R,getname(a));

% And give some output to the command line:
fprintf('\n%s\n\n',repmat('=',1,50));
a
S = average(100*R,3,'max1','dep');
show(S,'text','%4.1f');

