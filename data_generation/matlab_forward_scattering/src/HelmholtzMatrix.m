function H=HelmholtzMatrix(m,nx,ny,npml,h,fac,order,omega,type)

%  H = -(\triangle + \omega^2 I) 
n = nx*ny;

[sx sy sxp syp]= DistribPML(nx,ny,npml,fac);
Dxx1d = stiffness_matrix(nx,h,order);
Dyy1d = stiffness_matrix(ny,h,order);
Dx1d  = FirstOrderDifferenceMatrix1d(nx,h,order);
Dy1d  = FirstOrderDifferenceMatrix1d(ny,h,order);  
Dx    = kron(Dx1d,speye(ny));
Dy    = kron(speye(nx),Dy1d);
Dxx   = kron(Dxx1d,speye(ny));
Dyy   = kron(speye(nx),Dyy1d);
M     = spdiags(m(:),0,n,n);

switch type
    case 'compact'
        K  = Dxx + Dyy;
        Sx = spdiags(sx(:),0,n,n);
        Sy = spdiags(sy(:),0,n,n);
        H  = - omega^2*M...
             + I*omega*M*(Sx+Sy)...
             + M*Sx*Sy...
             - K...
             - Dx*(spdiags((sy(:)-sx(:))./(I*omega+sx(:)),0,n,n)*Dx)...
             - Dy*(spdiags((sx(:)-sy(:))./(I*omega+sy(:)),0,n,n)*Dy);
    case 'compact_explicit'
        H = - omega^2*M...
            +spdiags(-I/(omega*(npml-1)*h)*sxp(:)./(1-I/omega*sx(:)).^3,0,n,n)*Dx...
            +spdiags(-I/(omega*(npml-1)*h)*syp(:)./(1-I/omega*sy(:)).^3,0,n,n)*Dy...
            -spdiags(1./(1-I/omega*sx(:)).^2,0,n,n)*Dxx...
            -spdiags(1./(1-I/omega*sy(:)).^2,0,n,n)*Dyy;
    case 'auxiliary'
        K  = Dxx+Dyy;
        Sx = spdiags(sx(:),0,n,n);
        Sy = spdiags(sy(:),0,n,n);
        Id = speye(n);
        Mb = [M sparse(n,2*n);sparse(2*n,3*n)];
        Kb = [M*Sx*Sy-K -Dx -Dy;-(Sy-Sx)*Dx Sx sparse(n,n);-(Sx-Sy)*Dy sparse(n,n) Sy];
        Cb = [M*(Sx+Sy) sparse(n,2*n);sparse(n,n) Id sparse(n,n);sparse(n,2*n) Id];
        H  = (-omega^2*Mb+I*omega*Cb+Kb);
        
end