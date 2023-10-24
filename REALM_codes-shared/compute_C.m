    function Z = compute_C(U,X,Y)
        %% compute C_ij = ||U'(x_i - y_j)||^2;
        % U: d*k, X,Y: d*n; complexity is O(ndk)
        UtX = U'*X; UtY = U'*Y;
        temp1 = sum(UtX.^2,1)'; temp2 = sum(UtY.^2,1);
        Z = (-2*UtX')*UtY + temp1 + temp2;
        Z = max(Z,0);
    end