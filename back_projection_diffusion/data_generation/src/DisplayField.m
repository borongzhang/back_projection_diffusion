function DisplayField(u,x,y,npml)

nx=length(x);
ny=length(y);
h=x(2)-x(1);
u=reshape(u,ny,nx);

% if nargin==4
%     indx=npml+1:nx-npml;
%     indy=npml+1:ny-npml;
%     u=u(indy,indx);
%     x=x(indx);
%     y=y(indy);
% end
% figure
if isreal(u)
    set(gca, 'Fontsize', 12);
    imagesc(x,y,u);axis image;colorbar;
    if nargin==4
        hold on
        plot([x(1) x(end)],y(1)+(npml+1)*h*[1 1],'r');
        plot([x(1) x(end)],(y(end)-(npml+1)*h)*[1 1],'r');
        plot(x(1)+(npml+1)*h*[1 1],[y(1) y(end)],'r');
        plot((x(end)-(npml+1)*h)*[1 1],[y(1) y(end)],'r');
        hold off
    end
else
    subplot 211
    set(gca, 'Fontsize', 12);
    imagesc(x,y,real(u));axis image;colorbar;
    if nargin==4
        hold on
        plot([x(1) x(end)],y(1)+(npml+1)*h*[1 1],'r');
        plot([x(1) x(end)],(y(end)-(npml+1)*h)*[1 1],'r');
        plot(x(1)+(npml+1)*h*[1 1],[y(1) y(end)],'r');
        plot((x(end)-(npml+1)*h)*[1 1],[y(1) y(end)],'r');
        hold off
    end

    subplot 212
    set(gca, 'Fontsize', 12);
    imagesc(x,y,imag(u));axis image;colorbar;
    if nargin==4
        hold on
        plot([x(1) x(end)],y(1)+(npml+1)*h*[1 1],'r');
        plot([x(1) x(end)],(y(end)-(npml+1)*h)*[1 1],'r');
        plot(x(1)+(npml+1)*h*[1 1],[y(1) y(end)],'r');
        plot((x(end)-(npml+1)*h)*[1 1],[y(1) y(end)],'r');
        hold off
    end
end
