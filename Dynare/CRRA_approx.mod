var lny lnc lni lnk lnw r z h k y c w;
varexo eps;
parameters A cbeta alpha delta rho g gamma mu chi b;

A = 1;
cbeta = .995;
delta = .02;
alpha = .33;
g = 0;
rho = 0.9;
gamma = 2.2;
chi = .5259;
b = 1;
mu = 2.2863;

model;
z=rho*z(-1)+eps;
c=w*h+(1-delta+r)*k(-1)-(1+g)*k;
c^(-gamma)=cbeta*((1+g)*c(+1))^(-gamma)*(1+r(+1)-delta);
c^(-gamma)*w=chi*b*(1-h^mu)^((1-mu)/mu)*h^(mu-1);
y=A*k(-1)^alpha*(exp(z)*h)^(1-alpha);
w=(1-alpha)*y/h;
r=alpha*y/k(-1);
lny=log(y);
lnc=log(c);
lni=log(y-c);
lnk=log(k);
lnw=log(w);
end;

initval;
h=0.302706546347972;
z=0;
r=((1+g)^(gamma)/cbeta)-1+delta;
k=(A/r)^(1/(1-alpha))*h;
y=A*k^alpha*h^(1-alpha);
w=(1-alpha)*y/h;
c=w*h+(r-delta-g)*k;
end;

shocks;
var eps;
stderr 0.02;
end;

stoch_simul;