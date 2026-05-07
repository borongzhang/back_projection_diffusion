%% setup parameters
clear
addpath('src')

% Resolution of the perturbation n_{\eta}
N = 80;

% Dimension of the scattering data
Ntheta = N;

% Size of the scattering data
n = 500;

% Three frequencies of the data
freq_1 = 2.5;
freq_2 = 5.0;
freq_3 = 10.0;

% Directory to load the perturbation
dir_load = 'eta.h5';

% Directory to save the scattering data
dir_save = 'scatter.h5';

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

% order of accuracy
order = 8;

% intensity of the pml absorbtion
sigmaMax = 80;
%%
etas = h5read(dir_load,'/eta');

%%
% the number of angles of the sources and detectors (they are set to be the same).
dtheta = 2*pi/(Ntheta);

theta = linspace(pi, 3*pi-dtheta, Ntheta);
d = [cos(theta).' sin(theta).'];

theta_r = dtheta*(0:Ntheta-1);
r = [cos(theta_r).' sin(theta_r).'];

points_query = 0.5*r;
project_mat = zeros(Ntheta, nx, ny);

for ii = 1:nx
    for jj = 1:ny
        mat_dummy = zeros(nx,ny);
        mat_dummy(ii,jj) = 1;
        project_mat(:,ii,jj) = interp2(x,y,...
                                   reshape(mat_dummy, nx, ny),...
                                   points_query(:,1),...
                                   points_query(:,2));
    end
end

% properly reshaping and making it sparse
project_mat = sparse(reshape(project_mat, Ntheta, nx*ny));
 
scatter_real_freq_1 = zeros(Ntheta^2, n);
scatter_imag_freq_1 = zeros(Ntheta^2, n);
scatter_real_freq_2 = zeros(Ntheta^2, n);
scatter_imag_freq_2 = zeros(Ntheta^2, n);
scatter_real_freq_3 = zeros(Ntheta^2, n);
scatter_imag_freq_3 = zeros(Ntheta^2, n);
%%
omega1 = freq_1*2*pi;
omega2 = freq_2*2*pi;
omega3 = freq_3*2*pi;
U_in1 = exp(1i*omega1*(X(:)*d(:,1).'+ Y(:)*d(:,2).'));
U_in2 = exp(1i*omega2*(X(:)*d(:,1).'+ Y(:)*d(:,2).'));
U_in3 = exp(1i*omega3*(X(:)*d(:,1).'+ Y(:)*d(:,2).'));

%%   
for i = 1:n
    i
    eta = etas(:,i);
    
    m = 1 + eta;  
    eta_ext = ExtendModel(eta,nxi,nyi,npml);
    mext = ExtendModel(m,nxi,nyi,npml);
    
    H1 = HelmholtzMatrix(mext,nx,ny,npml,h,...
        sigmaMax,order,omega1,'compact_explicit');
    H2 = HelmholtzMatrix(mext,nx,ny,npml,h,...
        sigmaMax,order,omega2,'compact_explicit');
    H3 = HelmholtzMatrix(mext,nx,ny,npml,h,...
        sigmaMax,order,omega3,'compact_explicit');
    
    
    % building the right hand sides
    S1 = bsxfun(@times, -omega1^2*eta_ext, U_in1);
    S2 = bsxfun(@times, -omega2^2*eta_ext, U_in2);
    S3 = bsxfun(@times, -omega3^2*eta_ext, U_in3);
    
    % solving the equation
    U1 = H1\S1;
    U2 = H2\S2;
    U3 = H3\S3;
    
    % this is our "real data"
    scatter1 = project_mat*U1;
    scatter2 = project_mat*U2;
    scatter3 = project_mat*U3;
    
%     % adding noise
%     scatter = scatter.*(1 + randn(nxi,nyi));
    scatter_real_freq_1(:,i) = real(reshape(scatter1, Ntheta^2, 1));
    scatter_imag_freq_1(:,i) = imag(reshape(scatter1, Ntheta^2, 1));
    scatter_real_freq_2(:,i) = real(reshape(scatter2, Ntheta^2, 1));
    scatter_imag_freq_2(:,i) = imag(reshape(scatter2, Ntheta^2, 1));
    scatter_real_freq_3(:,i) = real(reshape(scatter3, Ntheta^2, 1));
    scatter_imag_freq_3(:,i) = imag(reshape(scatter3, Ntheta^2, 1));
end

%%
figure
DisplayField(scatter_imag_freq_2(:,8),xi,yi,0)

%%
h5create(dir_save,'/scatter_imag_freq_1',[Ntheta^2, n]);
h5create(dir_save,'/scatter_real_freq_1',[Ntheta^2, n]);
h5create(dir_save,'/scatter_imag_freq_2',[Ntheta^2, n]);
h5create(dir_save,'/scatter_real_freq_2',[Ntheta^2, n]);
h5create(dir_save,'/scatter_imag_freq_3',[Ntheta^2, n]);
h5create(dir_save,'/scatter_real_freq_3',[Ntheta^2, n]);

%%
h5write(dir_save, '/scatter_real_freq_1', scatter_real_freq_1);
h5write(dir_save, '/scatter_imag_freq_1', scatter_imag_freq_1);
h5write(dir_save, '/scatter_real_freq_2', scatter_real_freq_2);
h5write(dir_save, '/scatter_imag_freq_2', scatter_imag_freq_2);
h5write(dir_save, '/scatter_real_freq_3', scatter_real_freq_3);
h5write(dir_save, '/scatter_imag_freq_3', scatter_imag_freq_3);

