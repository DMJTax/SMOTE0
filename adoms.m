% generate new data in the adoms way

a = seldat(gendatb([10 0]),1);
n = size(a,1);
N = 300;
k = 3;

z = [];
for i=1:N
   I = ceil(rand(1,1)*n);
   x = +a(I,:);
   d = sqeucldistm(+a,x);
   [sd,J]=sort(d);
   a_pca = +a(J(1:(k+1)),:);
   sc = sd(ceil(rand(1,1)*k)+1); % 1NN is obj. itself
   [E,D]=eigs(cov(a_pca));
   z = [z; x+rand(1,1)*sc*E'];
end


figure(1); scatterd(a);
hold on;
scatterd(z,'ro');

