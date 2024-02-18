function U0 = generate_initial_U(seed,X,Y,r,c,nX,nY,k)
rng(seed);
K  = rand(nX,nY);
K  = max(0,K);
K  = K./sum(sum(K));
[K,~]  = Pi_post(K,r,c);
K2 = sum(K,2);
K1 = sum(K,1)';
MM = X*(K2.*X' - K*Y') + Y*(K1.*Y' - K'*X');
[V,~] = eigs(MM,k,'largestreal');
U0 = V;
end