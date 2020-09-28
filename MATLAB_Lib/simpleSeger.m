function [res] = simpleSeger(vec,tag)
    vec = vec(:);
    tag = tag(:);
    m = find(vec==tag);
    starter = m([2;m(2:end)-m(1:(end-1))]>1);
    ender = m([m(2:end)-m(1:(end-1));2]>1);
    res = [starter,ender];
end

