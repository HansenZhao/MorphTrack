function plotTagTrace(ha,vec,tag,cm,lw)
    % plotTagTrace(ha,vec,tag,cm)
    if ~exist('lw','var')
        lw = 1;
    end
    if ~exist('cm','var')
        cm = lines(7);
    end
    if isempty(ha)
        ha = gca;
    end
    
    [L,nc] = size(vec);
    if nc == 1
        vec = [(1:L)',vec];
        nc = 2;
    end
    
    taggers = unique(tag);
    nk = length(taggers);
    
    
    if nc == 2
        plotFunc = @(ha,vec,c)plot(ha,vec(:,1),vec(:,2),'LineWidth',lw,'Color',c);
    elseif nc == 3
        plotFunc = @(ha,vec,c)plot3(ha,vec(:,1),vec(:,2),vec(:,3),'LineWidth',lw,'Color',c);
    end
    
    plotFunc(ha,vec,[0.8,0.8,0.8]);
    hold on;
    for m = 1:1:nk
        if ~isnan(taggers(m))
            segs = simpleSeger(tag,taggers(m));
            for n = 1:1:size(segs,1)
                plotFunc(ha,vec(segs(n,1):segs(n,2),:),cm(taggers(m),:));
            end
        end
    end
end

