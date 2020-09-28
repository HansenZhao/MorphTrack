function [estV,realV,corr,mre] = edm(vecFrom,vecEst,e,tau,isShow)
    if nargin == 4
        isShow = 1;
    end
    L = length(vecFrom);
    if L ~= length(vecEst)
        error('EDM: vector length of origin and target should identical!');
    end
    newL = L - (e-1)*tau;
    oriMat = zeros(newL,e);
    tarMat = zeros(newL,e);
    for m = 1:1:e
        starter = 1+(e-m)*tau; %1 + (e-1)*tau - (m-1)*tau
        ender = L - (m-1)*tau;
        oriMat(:,m) = vecFrom(starter:ender);
        tarMat(:,m) = vecFrom(starter:ender);
    end
    realV = vecEst((1+(e-1)*tau):end);
    estV = zeros(newL,1);
    for m = 1:1:newL
        [D,I] = pdist2(oriMat,oriMat(m,:),'euclidean','Smallest',e+2);
        D(1) = []; I(1) = [];
        u = exp(-D./D(1));
        w = u./sum(u);
        estV(m) = w' * realV(I);
    end
    I = ~isnan(estV);
    estV = estV(I); realV = realV(I);
    corr = corrcoef([estV,realV]);
    corr = corr(2);
    I = realV ~= 0;
    mre = mean((estV(I) - realV(I))./realV(I));
    if isShow
        figure;
        scatter(estV,realV,5,'filled'); box on;
        xlabel('estimate Y'); ylabel('real Y');
    end
end

