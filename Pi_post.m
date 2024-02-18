function [Pi,feasi] = Pi_post(Pi,r,c)
% The rounding procedure given in Algorithm 2 in the following paper:
% J. Altschuler, J. Niles-Weed, and P.Rigollet, Near-linear time approximation algorithms for optimal transport via
%  Sinkhorn iteration, Advances in Neural Information Processing Systems, 30: 1961–1971, 2017.
[nX,nY] = size(Pi);
ones_nY = ones(nY,1);
ones_nX = ones(nX,1);
Pi2 = Pi*ones_nY;
x = min(r./Pi2,1);
Pi = Pi.*x;
Pi1 = Pi'*ones_nX;
y = min(c./Pi1,1);
Pi = Pi.*y';
err_r = r - Pi*ones_nY; %sum(Pi,2);
err_c = c - Pi'*ones_nX;
sum_err_c = sum(err_c);
if sum_err_c ~= 0
Pi = Pi + err_r*err_c'/sum(err_c);
end
feasi = norm(Pi*ones_nY - r) + norm(Pi'*ones_nX - c);