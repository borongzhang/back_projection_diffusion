%% setup scaling parameters
clear
addpath('src')

% Resolution of the perturbation n_{\eta}
N = 80;

% Number of the perturbations
n = 500;

%% setup the model and the domain

h =  1/(N-1);
% background wavespeed
c = ones(N,N);

% size of the model in interior domain
nxi  = size(c,2);
nyi  = size(c,1);
ni   = nxi*nyi;

xi = h*(0:nxi-1) - 0.5;
yi = h*(0:nyi-1) - 0.5;

[Xi,Yi] = meshgrid(xi,yi);

% size of the simulation domain
npml = 20;
nx = nxi + 2*npml;
ny = nyi + 2*npml;

x  = [xi(1)+(-npml:-1)*h xi xi(end)+(1:npml)*h];
y  = [yi(1)+(-npml:-1)*h yi yi(end)+(1:npml)*h];

[X,Y] = meshgrid(x,y);

%%
ns_max = 20;
ns = 20;
scatter_width_per_mesh = 10;
delta_m = 0.2;


etas = zeros(N^2, n);

%% 
for i = 1:n
    r_tmp = 0.35*sqrt(rand(ns_max, 1)); % scale by inverse r^2 for uniform in S(0,1)
    theta_tmp = 2*pi*rand(ns_max, 1);
    centres = diag(r_tmp)*[cos(theta_tmp), sin(theta_tmp)];
    for ii = 1:ns
        Xi_tmp = Xi(:)-centres(ii,1);
        Yi_tmp = Yi(:)-centres(ii,2);
        eta_new = abs(Xi_tmp)<=0.5*scatter_width_per_mesh*h;
        eta_new = eta_new.*(abs(Yi_tmp)<=0.5*scatter_width_per_mesh*h);
        eta_new = delta_m*eta_new;
    
        if ii == 1
            eta = eta_new;
        else
            eta = eta + eta_new;
        end
    end 
    etas(:,i) = reshape(eta, N^2, 1);
end

%%
figure
DisplayField(etas(:,6),xi,yi,0)

%%
h5create('eta.h5','/eta',[N^2, n]);
h5write('eta.h5','/eta', etas);
