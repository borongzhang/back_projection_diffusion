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
ns_max = 10;
delta_m = 0.2;
etas = zeros(N^2, n);

%%
for i = 1:n
    r_tmp = 0.35*sqrt(rand(ns_max, 1)); % scale by inverse r^2 for uniform in S(0,1)
    theta_tmp = 2*pi*rand(ns_max, 1);
    centres = diag(r_tmp)*[cos(theta_tmp), sin(theta_tmp)];
    ns = randi(ns_max);
    %generate a random Shepp-Logan phantom.
    for ii = 1:ns
        width = randi(3);
        if width==1
            scatter_width_per_mesh = 3;
        elseif width==2
            scatter_width_per_mesh = 5;
        elseif width==3
            scatter_width_per_mesh = 10;
        end
        eta_new = 0*Xi;
        col_ind = find(Xi(1,:)>centres(ii,1),1);
        row_ind = find(Yi(:,1)>centres(ii,2),1);
        % randomly choose rotation
        rot = randi(4);
        if rot==1
            for pp = 1:scatter_width_per_mesh
                eta_new(row_ind+(pp-1), col_ind+(pp:scatter_width_per_mesh))=1;
            end
        elseif rot==2
            for pp = 1:scatter_width_per_mesh
                eta_new(row_ind-(pp-1), col_ind+(pp:scatter_width_per_mesh))=1;
            end
        elseif rot==3
            for pp = 1:scatter_width_per_mesh
                eta_new(row_ind+(pp-1), col_ind+(1:(scatter_width_per_mesh-pp+1)))=1;
            end
        elseif rot==4
            for pp = 1:scatter_width_per_mesh
                eta_new(row_ind-(pp-1), col_ind+(1:(scatter_width_per_mesh-pp+1)))=1;
            end
        end
        eta_new = delta_m*eta_new(:);

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
DisplayField(etas(:,8),xi,yi,0)

%%
h5create('eta.h5','/eta',[N^2, n]);
h5write('eta.h5','/eta', etas);

