function [ b,I ] = torIsMember( vec,content,tor )
    if ~exist('tor','var')
        tor = 0.01;
    end
    diff = abs(vec(:)-content(:)');
    [d,I] = min(diff,[],2);
    b = d<tor;
    I(d>tor) = -1;
end

