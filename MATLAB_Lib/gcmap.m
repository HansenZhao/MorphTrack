function [ cm ] = gcmap(c3,c1,c2)
    if nargin == 1
        cm = [linspace(1,c3(1),64)',linspace(1,c3(2),64)',linspace(1,c3(3),64)'];
    else
        if nargin < 3
            c2 = [1,1,1];
        end
        cm = [linspace(c1(1),c2(1),32)',linspace(c1(2),c2(2),32)',linspace(c1(3),c2(3),32)';...
              linspace(c2(1),c3(1),32)',linspace(c2(2),c3(2),32)',linspace(c2(3),c3(3),32)'];
    end
    
end

