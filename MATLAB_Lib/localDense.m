function [dense] = localDense(pos,r)
    L = size(pos,1);
    dist = pdist2(pos,pos,'euclidean');
    dense = sum(dist < r,2)-1;
end

