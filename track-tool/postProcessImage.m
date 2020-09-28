function [L] = postProcessImage(rawMat)
    rawMat = imfill(rawMat,'holes');
    rawMat = imopen(rawMat,strel('disk',2));
    rawMat = bwareaopen(rawMat,25);
    rawMat = imclearborder(rawMat,4);
    cc = bwconncomp(rawMat,8);
    L = labelmatrix(cc);
    L = imdilate(L,strel('disk',1));
    L = L > 0;
end

