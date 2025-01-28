function mnew=ExtendModel(m,nxint,nyint,npml)

m=reshape(m,nyint,nxint);

nx=size(m,2)+2*npml;
ny=size(m,1)+2*npml;

mnew=zeros(ny,nx);
mnew(npml+1:ny-npml,npml+1:nx-npml)=m;
mnew(1:npml,:)=repmat(mnew(npml+1,:),npml,1);
mnew(end-npml+1:end,:)=repmat(mnew(end-npml,:),npml,1);
mnew(:,1:npml)=repmat(mnew(:,npml+1),1,npml);
mnew(:,end-npml+1:end)=repmat(mnew(:,end-npml),1,npml);
mnew=mnew(:);