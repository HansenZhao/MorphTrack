function [ d,mat ] = MSMDist( vec1,vec2,c )
    %Bagnall, A.; Lines, J.; Bostrom, A.; Large, J.; Keogh, E. 
    %Data Mining and Knowledge Discovery 2017, 31, 606-660.
    L = length(vec1);
    if length(vec2) ~= L
        error('inconsist vector length!');
    end
    mat = zeros(L);
    mat(1) = abs(vec1(1)-vec2(1));
    for m = 2:L
        mat(m,1) = mat(m-1,1) + MSMcost(vec1(m),vec1(m-1),vec2(1),c);
    end
    for m = 2:L
        mat(1,m) = mat(1,m-1) + MSMcost(vec2(m),vec1(1),vec2(m-1),c);
    end
    for m = 2:L
        for n = 2:L
            mat(m,n) = min([mat(m-1,n-1)+abs(vec1(m)-vec2(n)),...
                mat(m-1,n)+MSMcost(vec1(m),vec1(m-1),vec2(n),c),...
                mat(m,n-1)+MSMcost(vec2(n),vec1(m),vec2(n-1),c)]);
        end
    end
    d = mat(L,L);
end

function cost = MSMcost(a,a2,b,c)
    if (a<=a2 && a2<=b) || (a>=a2 && a2>=b)
        cost = c;
    else
        cost = c + min([abs(a-a2),abs(a2-b)]);
    end
end

