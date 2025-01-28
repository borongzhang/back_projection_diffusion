function Dx=FirstOrderDifferenceMatrix1d(nx,h,order)
% order is for the order of accuracy

Dx=spdiags(repmat(FDweights(order/2,0:order,1).'/h,nx,1),-(order/2):(order/2),nx,nx);
for i=1:((order/2)-1)
    weights=FDweights(i,0:order+2,1).'/h;
    Dx(i,1:order+2)=weights(2:end);

    weights=FDweights(order+2-i,0:order+2,1).'/h;
    Dx(end-(i-1),(end-(order+1)):end)=weights(1:end-1);
end