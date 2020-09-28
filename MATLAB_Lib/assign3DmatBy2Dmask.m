function [ maskedMat ] = assign3DmatBy2Dmask(mat3,mask,value)
    %[ maskedMat ] = assign3DmatBy2Dmask(mat3,mask,value)
    L = size(mat3,3);
    if length(value(:))>2
       for m = 1:L
           tmp = mat3(:,:,m);
           tmp(mask) = value(mask);
           mat3(:,:,m) = value
       end
    else
    end
    
    
end

