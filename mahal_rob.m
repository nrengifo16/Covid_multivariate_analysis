function d = mahal_rob(X, type)
    if nargin < 2
        type='Kendall';
    end
    [m, n] = size(X);
    
    covX = zeros(n,n);

    corrX = corr(X, 'Type', type);
    
    % Median Absolute Deviation
    madX = mad(X, 1);

    for j=1:n
        for i=1:n
            covX(i,j) = corrX(i,j)*madX(i)*madX(j);
        end
    end
    
    invCovX = pinv(covX);
    
    centeredX = (X - mean(X));
    
    for k=1:m
        d(k) = centeredX(k,:)*invCovX*centeredX(k,:)';
    end

    d = d';

end