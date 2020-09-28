function [ pkLocs,pksIntens,I ] = hsRoughFindPeaks2( locs,profiles,halfAcc,minTor,isShow,ha )
    if ~exist('isShow','var')
        isShow = 1;
    end
    if ~exist('minTor','var')
        minTor = 0.01;
    end
    if ~exist('ha','var')
        figure;
        ha = gca;
    end
    pkLocs = matList();
    pksIntens = matList();
    I = matList();
    torIntens = max(profiles) * minTor;
    L = length(locs);
    indices = 1:L;
    m = 1;
    while(m<=L)
        dist = abs(locs - locs(m));
        bInRange = dist <= halfAcc;
        [intens,locId] = max(profiles(bInRange));
        idsInRange = indices(bInRange);
        if intens > torIntens
            maxId = idsInRange(locId);
            if maxId < m
                m = max([m+1,idsInRange(end)]);
            elseif maxId > m
                m = maxId;
            else
                I.addOne(m);
                pkLocs.addOne(locs(m));
                pksIntens.addOne(profiles(m));
                m = max([m+1,idsInRange(end)]);
            end
        else
            m = max([m+1,idsInRange(end)]);
        end        
    end
    pkLocs = cell2mat(pkLocs.data);
    pksIntens = cell2mat(pksIntens.data);
    I = cell2mat(I.data);
    if isShow
        plot(ha,locs,profiles);
        ha.NextPlot = 'add';
        scatter(ha,pkLocs,pksIntens,'filled');
        ha.NextPlot = 'replace';
    end
end

