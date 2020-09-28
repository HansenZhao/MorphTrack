function [ conMatCell,simiIndexCell ] = ccluster(X,clusterFunc,K,sampleRate,iterTime)
    % I = clusterFunc(X,k,option);
    % Monti, S.; Tamayo, P.; Mesirov, J.; Golub, T., Consensus Clustering: 
    % A Resampling-Based Method for Class Discovery and Visualization of Gene 
    % Expression Microarray Data. Machine Learning 2003, 52 (1), 91-118.
    [sampNum,feaNum] = size(X);
    [conMatCell,simiIndexCell] = deal(cell(length(K),1));
    for n = 1:length(K)
        [I,M] = deal(zeros(sampNum,sampNum,iterTime));
        for m = 1:1:iterTime
            sampIndex = randsample(1:sampNum,round(sampNum*sampleRate));
            [rI,cI] = meshgrid(1:sampNum);
            I(:,:,m) = and(ismember(rI,sampIndex),ismember(cI,sampIndex));
            feaIndex = randsample(1:feaNum,round(feaNum*sampleRate));
            x = X(sampIndex,feaIndex);
            cRes = clusterFunc(x,K(n));
            tmp = zeros(sampNum,1);
            tmp(sampIndex) = cRes;
            M(:,:,m) = and(tmp(rI)==tmp(cI),I(:,:,m)>0);
        end
        conMat = sum(M,3)./(sum(I,3)+eps);
        simiIndex = kmeans(conMat,K(n));
        
        [~,sortIndex] = sort(simiIndex);
        
        figure; ha = gca;
        imagesc(conMat(sortIndex,sortIndex));colormap(gcmap([1,0,0]));
        ha.CLim = [0,1];
        title(sprintf('K = %d',K(n)));
        conMatCell{n} = conMat;
        simiIndexCell{n} = simiIndex;
    end  
end

