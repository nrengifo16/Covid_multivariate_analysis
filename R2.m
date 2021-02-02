function r2 = R2(yhat,ye)
[n ~] = size(ye);
res = ye-yhat;
SSE = sum(res.^2);
SST = n*var(ye);
r2 = 1-SSE/SST;
end