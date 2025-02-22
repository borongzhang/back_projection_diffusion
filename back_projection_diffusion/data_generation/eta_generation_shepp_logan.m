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
pad = 5;
% we define the gaussian smoother
gaus = exp(-(Xi.^2 + Yi.^2)/0.00025);% we smoothen the random field using a convolution
           
window = exp(-0.005./( 0.2304-(Xi.^2+Yi.^2))).*(sqrt(Xi.^2+Yi.^2)<0.48);
window(isnan(window)) = 0;

etas = zeros(N^2, n);

%%   
for i = 1:n
    %generate a random Shepp-Logan phantom.
    eta = random_Shepp_Logan(nxi-2*pad,{'pad', pad; 'M', 1; 'phantom','msl'});
    eta = reshape(eta, nxi, nyi);
    smooth = conv2(gaus, eta, 'same'); 
    smooth = smooth/max(max(abs(smooth)));
    eta = smooth.*window;
    etas(:,i) = reshape(eta, N^2, 1);
end


%%
figure
DisplayField(etas(:,8),xi,yi,0)

%%
h5create('eta.h5','/eta',[N^2, n]);
h5write('eta.h5','/eta', etas);


