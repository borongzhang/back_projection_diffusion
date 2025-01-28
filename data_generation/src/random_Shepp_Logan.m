function images = random_Shepp_Logan(n,param)
%
% function images = randomSheppLogan(n,param)
%
% Authors:
%   (c) Matthias Chung (e-mail: mcchung@vt.edu)               in April 2017
%        
% MATLAB Version: 9.0.0.341360 (R2016a)
%
% Description:
%   Generates randomized versions of Shepp-Logan phantom. This code bases
%   on Matlabs phantom function.
%
% Input arguments:
%   n         - nxn size of image 
%   #param    - further options of algorithm
%     phantom - if 'sl' or 'msl'(default) standard Shepp-Logan or Modified 
%               Shepp-Logan will be loaded otherwise input needs to be an
%               m x 6 matrix with parameters (density, ellipse length 1,
%               ellipse length 2, center x, center y, and rotation angle)
%     default - if true then default Shepp-Logan will be loaded (default false)
%     M       - number of random images (default 1)
%     pad     - padding the of the image (default 0)
%
% Output arguments:
%   images    - each column defines one nxn random image
%
% Details:
%
% Example:
%    >> randomSheppLogan; % will plot a random image of size 512 x 512
%    >> P = randomSheppLogan(256,{'pad', 4; 'M', 2; 'phantom','sl'});
%    >> figure, imagesc(reshape(P(:,1),264,264))
%
% References:
%   [1] A. K. Jain, "Fundamentals of Digital Image Processing", Prentice-Hall, 1989, p. 439.
%   [2] Shepp, Larry A.; Logan, Benjamin F. (June 1974). "The Fourier 
%       Reconstruction of a Head Section"  (PDF). IEEE Transactions on 
%       Nuclear Science. NS-21 (3): 21--43.
%   [3] Matlab documentation on phantom function
%   [4] Lars Ruthotto, Julianne Chung, Matthias Chung, "Optimal
%       Experimental Design for Constrained Inverse Problems", ArXiV.org
%

% default settings
pad = 4;                                                       % padding outside
M   = 1;                                                       % number of generated images
default = false;                                               % default if true it will generate non randomized phantoms
if ~nargin, n = 512; end

% rewrite default parameters if needed
if nargin == nargin(mfilename)
  for j = 1:size(param,1), eval([param{j,1},'= param{j,2};']); end
end

if ~exist('phantom','var'), phantom = sheppLogan('msl'); end
if ischar(phantom), phantom = sheppLogan(phantom); end

% set fixed parameters
images  = zeros((n+2*pad)^2,M);                         % initialize images
pix     = linspace(-1,1,n);                             % range of x and y 
[X,Y]   = meshgrid(pix,fliplr(pix));                    % generate meshgrid
if pad, z1 = zeros(n+2*pad,pad); z2 = zeros(pad,n); end % add padding if requested

for m = 1:M
  
  if default % use standard Shepp-Logan phantom
    rphantom = phantom;
  else
    rphantom = modify(phantom); % get random variation of phantom
  end
  
  image = generateImage(rphantom,n,X,Y); % generate image
  
  % if padding is requested
  if pad, image = [z1, [z2; image; z2], z1]; end
   
  if ~nargin, figure, imagesc(image), colorbar, end
  
  images(:,m) = image(:); % record images
end


end

% -------------------------------------------------------------------------
% subroutine on generating figure values ----------------------------------
function image = generateImage(e,n,X,Y)

image  = zeros(n);        % initialize p
e(:,2) = e(:,2).^2;       % update elements in phantom a^2
e(:,3) = e(:,3).^2;       % update elements in phantom b^2
e(:,6) = e (:,6)*pi/180;  % convert angle to radians
cosp   = cos(e(:,6));     % take cosine of angles
sinp   = sin(e(:,6));     % take sine of angles

for k = 1:size(e,1) 
                          % correct by center of ellipse
  x = X - e(k,4);         % x offset
  y = Y - e(k,5);         % y offset
  
                          % define ellipse
  ellipse = ((x*cosp(k) + y*sinp(k)).^2)/e(k,2) + ((y*cosp(k) - x*sinp(k)).^2)/e(k,3);
  
  idx = find( ellipse <= 1);          % pixel index within ellipse
  if k < 5
    image(idx) = e(k,1);              % set density in ellipse
  else
    image(idx) = image(idx) + e(k,1); % add density in ellipse to pixel values 
  end
  
end

end

% -------------------------------------------------------------------------
% subroutine on modifying the phantom -------------------------------------
function phantom = modify(phantom)

m = size(phantom,1); % number of ellipes

% generate random scaling
scale = 1-rand*2/9; phantom(:,2:5) = scale*phantom(:,2:5);

% random rotation
rotation = 2*45*(rand-0.5); phantom(:,6) = rotation + phantom(:,6);

% random translation
translate = 0.2*rand(1,2); phantom(:,4:5) = repmat(translate,m,1) + phantom(:,4:5);

% randomize density (relative to density and within density [0,1])
density = 2*0.1*(rand(m,1)-0.5); phantom(:,1) = density.*phantom(:,1)+phantom(:,1);
phantom(:,1)  = bsxfun(@min,bsxfun(@max,phantom(:,1),0),1); % ensure densities between [0,1]

% remove randomly tumors
obj = 4; idx = randsample(m-obj,randi(m-obj)-1,'false'); phantom(idx+obj,:) = [];

end

% -------------------------------------------------------------------------
% subroutine on loading default phantoms ----------------------------------
function phantom = sheppLogan(type)
%
%     Column 1:  A    the additive intensity value of the ellipse
%     Column 2:  a    the length of the horizontal semi-axis of the ellipse
%     Column 3:  b    the length of the vertical semi-axis of the ellipse
%     Column 4:  x0   the x-coordinate of the center of the ellipse
%     Column 5:  y0   the y-coordinate of the center of the ellipse
%     Column 6:  phi  the angle (in degrees) between the horizontal semi-axis
%                     of the ellipse and the x-axis of the image

if strcmp(type,'sl')
  %
  %  Default head phantom, taken from AK Jain, 439.
  %
  %   A       a      b       x0      y0    phi
  %  ---------------------------------------------
  phantom = ...
    [   1    0.69    0.92      0       0    0   % outer skull cap boundaries
     0.02  0.6624  0.8740      0  -.0184    0   % inner skull cap boundaries (intensity corrected since not adding)
     0.00  0.1100  0.3100   0.22       0  -18   % ventricle left (intensity corrected since not adding)
     0.00  0.1600  0.4100  -0.22       0   18   % ventricle right (intensity corrected since not adding)
     0.01  0.2100  0.2500      0    0.35    0   % tumor 1
     0.01  0.0460  0.0460      0     0.1    0   % tumor 2
     0.01  0.0460  0.0460      0    -0.1    0   % tumor 3
     0.01  0.0460  0.0230  -0.08  -0.605    0   % tumor 4
     0.01  0.0230  0.0230      0  -0.606    0   % tumor 5
     0.01  0.0230  0.0460   0.06  -0.605    0]; % tumor 6
elseif strcmp(type,'msl')
  %
  %   This head phantom is the same as the Shepp-Logan except
  %   the intensities are changed to yield higher contrast in
  %   the image.  Taken from Toft, 199-200.
  %
  %   A       a      b       x0      y0    phi
  %  --------------------------------------------- 
  phantom = ...
    [  1    0.69    0.92      0        0    0   % outer skull cap boundaries
     0.2  0.6624  0.8740      0  -0.0184    0   % inner skull cap boundaries (intensity corrected since not adding)
     0.0  0.1100  0.3100   0.22        0  -18   % ventricle left (intensity corrected since not adding)
     0.0  0.1600  0.4100  -0.22        0   18   % ventricle right (intensity corrected since not adding)
     0.1  0.2100  0.2500      0     0.35    0   % tumor 1
     0.1  0.0460  0.0460      0      0.1    0   % tumor 2
     0.1  0.0460  0.0460      0     -0.1    0   % tumor 3
     0.1  0.0460  0.0230  -0.08   -0.605    0   % tumor 4
     0.1  0.0230  0.0230      0   -0.606    0   % tumor 5
     0.1  0.0230  0.0460   0.06   -0.605    0]; % tumor 6    
   
else
  error('No phantom selected.')
end

end