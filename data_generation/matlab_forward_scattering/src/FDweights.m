function cwei = FDweights(z,x,m)

%---------------------------------
% finite-difference weights 
% (Fornberg algorithm) 
%
% z:  expansion point 
% x:  vector of evaluation points 
% m:  order of derivative 
%
% Example: cwei = FDweights(0,[0 1 2],1); 
% gives    cwei = [-3/2  2  -1/2]
%
% h f'_0 = -3/2 f_0 + 2 f_1 - 1/2 f_2 
%
%---------------------------------

  n  = length(x)-1;
  c1 = 1;
  c4 = x(1)-z;
  c = zeros(n+1,m+1);
  c(1,1) = 1;
  for i=1:n
    mn = min(i,m);
    c2 = 1;
    c5 = c4;
    c4 = x(i+1)-z;
    for j=0:i-1
      c3 = x(i+1)-x(j+1);
      c2 = c2*c3;
      if (j == (i-1)) 
        for k=mn:-1:1
          c(i+1,k+1) = c1*(k*c(i,k)-c5*c(i,k+1))/c2;
        end
        c(i+1,1) = -c1*c5*c(i,1)/c2;
      end
      for k=mn:-1:1
        c(j+1,k+1) = (c4*c(j+1,k+1)-k*c(j+1,k))/c3;
      end
      c(j+1,1) = c4*c(j+1,1)/c3;
    end
    c1 = c2;
  end
  cwei = c(:,end);
