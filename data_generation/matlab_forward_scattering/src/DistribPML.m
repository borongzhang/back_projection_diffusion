function [sigmaX sigmaY sigmaXp sigmaYp]=DistribPML(nx,ny,nPML,fac)

t=linspace(0,1,nPML);

sigmaX=zeros(ny,nx);
sigmaX(:,1:nPML)=repmat(fac*t(end:-1:1).^2,ny,1);
sigmaX(:,nx-nPML+1:nx)=repmat(fac*t.^2,ny,1);

sigmaY=zeros(ny,nx);
sigmaY(1:nPML,:)=repmat(fac*t(end:-1:1)'.^2,1,nx);
sigmaY(ny-nPML+1:ny,:)=repmat(fac*t'.^2,1,nx);

sigmaXp=zeros(ny,nx);
sigmaXp(:,1:nPML)=repmat(-2*fac*t(end:-1:1),ny,1);
sigmaXp(:,nx-nPML+1:nx)=repmat(2*fac*t,ny,1);

sigmaYp=zeros(ny,nx);
sigmaYp(1:nPML,:)=repmat(-2*fac*t(end:-1:1)',1,nx);
sigmaYp(ny-nPML+1:ny,:)=repmat(2*fac*t',1,nx);