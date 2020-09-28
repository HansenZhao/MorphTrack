function [ I,res ] = DPCluster( X,cutOffDis,method)
    %Rodriguez, A.; Laio, A., Clustering by fast search and find of density peaks. 
    %Science 2014, 344 (6191), 1492-1496.UNTITLED2 Summary of this function goes here
    %Return the local density of all points
    [locDens,disMat] = localDens(X,cutOffDis,method);
    hpDis = higherPeakDis(locDens,disMat);
    res = [locDens',hpDis];
    hf = figure;
    scatter(locDens,hpDis,'filled'); drawnow;
    rect = getrect();
    subX = locDens'-rect(1); subY = hpDis-rect(2);
    I = find(and(and(subX>0,subX<rect(3)),and(subY>0,subY<rect(4))));
    hold on;
    scatter(locDens(I),hpDis(I),'filled');
    figure;
    xy = tsne(X);
    scatter(xy(:,1),xy(:,2),'filled'); hold on; scatter(xy(I,1),xy(I,2),'filled');
end

function [ hpDis ] = higherPeakDis(locDens,disMat)
    L = length(locDens);
    hpDis = zeros(L,1);
    for m = 1:1:L
        tmp = min(disMat(m,(locDens - locDens(m))>0));
        if tmp
            hpDis(m) = tmp;
        else
            hpDis(m) = max(disMat(m,:));
        end
    end
end



function [ locDens,disMat ] = localDens( X,cutOffDis,method)
    %Rodriguez, A.; Laio, A., Clustering by fast search and find of density peaks. 
    %Science 2014, 344 (6191), 1492-1496.UNTITLED2 Summary of this function goes here
    %Return the local density of all points
    disMat = pdist2(X,X,method);
    fprintf('Distance Matrix max: %.3f, min: %.3f\n',max(disMat(:)),min(disMat(:)));
    locDens = sum((disMat-cutOffDis)<=0);
end

