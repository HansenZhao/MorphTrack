function [ res ] = eliMaxMin( mat )
    [r,c] = size(mat);
    res = zeros(r-2,c);
    for m = 1:c
        tmp = mat(:,m);
        [~,I] = max(tmp);
        tmp(I) = [];
        [~,I] = min(tmp);
        tmp(I) = [];
        res(:,m) = tmp;
    end
end

