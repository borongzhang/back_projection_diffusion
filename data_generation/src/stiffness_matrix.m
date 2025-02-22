function D_xx = stiffness_matrix(N_x, dx, order_x)
% 1D finite difference matrix (Dirichlet boundary nodes are not on the grid)
% switch order_x
%     case 2 
%         D_xx = spdiags( repmat(FDweights(1,0:2,2).', N_x,1), [-1 0 1], N_x, N_x) /dx^2;
%     case 4
%         D_xx = spdiags( repmat( FDweights(2,0:4,2).' ,N_x,1), [-2 -1 0 1 2], N_x,N_x)/ dx^2;
%         weights = FDweights(1,0:6,2).'/dx^2;
%         D_xx(1, 1:6) = weights(2:end);
%         weights = FDweights(5,0:6,2).'/dx^2;
%         D_xx(end, end-5:end ) = weights(1:end-1);
%          %% have to complete with higher order matrices
%     case 6
%         D_xx = spdiags( repmat( FDweights(3,0:6,2).' ,N_x,1), [-3 -2 -1 0 1 2 3], N_x,N_x)/ dx^2;
%         weights = FDweights(1,0:8,2).'/dx^2;
%         D_xx(1,1:8) = weights(2:end);
%         weights = FDweights(7,0:8,2).'/dx^2;
%         D_xx(end, end-7:end ) = weights(1:end-1);
%         
% 
%         weights = FDweights(2,0:8,2).'/dx^2;
%         D_xx(2,1:8) = weights(2:end);
%         weights = FDweights(6,0:8,2).'/dx^2;
%         D_xx(end-1, end-7:end ) = weights(1:end-1);
% 
% end


%%
D_xx=spdiags(repmat(FDweights(order_x/2,0:order_x,2).'/dx^2,N_x,1),-(order_x/2):(order_x/2),N_x,N_x);
for i=1:((order_x/2)-1)
    weights=FDweights(i,0:order_x+2,2).'/dx^2;
    D_xx(i,1:order_x+2)=weights(2:end);

    weights=FDweights(order_x+2-i,0:order_x+2,2).'/dx^2;
    D_xx(end-(i-1),(end-(order_x+1)):end)=weights(1:end-1);
end