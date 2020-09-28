function [ res,statis ] = hsAnova1(x,groupTag,isSpeak)
    L = length(x);
    if length(groupTag) ~= L
        disp('ANOVA: sample length should be same as tag length!');
        return;
    end
    if ~exist('isSpeak','var')
        isSpeak = 1;
    end
    tagSet = unique(groupTag);
    k = length(tagSet);
    if k <= 1
        disp('ANOVA: group tag should contain more than one groups');
        return;
    end
    %% Sum of Squares  
    ssb = 0;
    ssw = 0;
    %% group statistic
    groupLength = zeros(k,1);
    groupMean = zeros(k,1);
    groupVar = zeros(k,1);
    %% loop calculation
    for m = 1:1:k
        groupX = x(groupTag == tagSet(m));
        ssb = ssb + length(groupX) * power(mean(groupX)-mean(x),2);
        ssw = ssw + sum((groupX - mean(groupX)).^2);
        groupLength(m) = length(groupX);
        groupMean(m) = mean(groupX);
        groupVar(m) = var(groupX);
    end 
    sumSquares = [ssb,ssw,ssb+ssw]';
    %% Degree of Freedom
    degreeFreedom = [k-1,L-k,L-1]';
    %% Mean of Square
    meanSqures = sumSquares./degreeFreedom;
    meanSqures(3) = nan;
    %% F statistic
    Fstatistic = nan(3,1);
    Fstatistic(1) = meanSqures(1)/meanSqures(2);
    %% P value
    Pvalue = nan(3,1);
    Pvalue(1) = 1 - fcdf(Fstatistic(1),degreeFreedom(1),degreeFreedom(2));
    %% table
    res = table(sumSquares,degreeFreedom,meanSqures,Fstatistic,Pvalue,'RowNames',...
                {'Between','Within','Total'});
    statis = table(tagSet(:),groupLength,groupMean,groupVar,'VariableNames',...
                  {'tag','length','mean','variance'});
    %% disp
    if isSpeak
        disp(statis);
        disp(res);
    end
end

