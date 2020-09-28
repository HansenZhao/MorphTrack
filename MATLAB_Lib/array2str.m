function [ str ] = array2str(vec) 
    vec = vec(:)';
    str = num2str(vec);
    str = strrep(str,' ',',');
    I1 = [0,(str(2:end)-str(1:(end-1)))==0];
    I2 = str==',';
    str(and(I1,I2)) = [];
end

