function [ acc,matchRes,costMat ] = matchAcc( gt,pred )
    gt = gt(:);
    pred = pred(:);
    if length(gt) ~= length(pred)
        error('the length of array is not consistent!')
    end
    gtTag = unique(gt);
    predTag = unique(pred);
    nTag = length(gtTag);
    if length(predTag) > nTag
        error('the number of tags in gt and pred is not consistent!')
    elseif length(predTag)<nTag
        predTag = [predTag(:);max(predTag)+(1:(nTag-length(predTag)))'];
    end
    costMat = zeros(nTag);
    for m = 1:nTag
        for n = 1:nTag
            costMat(m,n) = sum(and(gt==gtTag(m),pred==predTag(n)));
        end
    end
    res = lapjv(max(costMat(:))-costMat);
    matchRes = [gtTag(:),predTag(res(:))];
    pred2 = zeros(size(pred));
    for m = 1:nTag
        I = pred==matchRes(m,2);
        pred2(I) = matchRes(m,1);
    end
    acc = sum(pred2==gt)/length(gt);
end

