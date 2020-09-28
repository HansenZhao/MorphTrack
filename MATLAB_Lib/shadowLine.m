function shadowLine(x,mat,color,ha)
    if ~exist('ha','var')
        ha = gca;
    end
    x = x(:)';
    y = mean(mat);
    sy = std(mat);
    ha.NextPlot = 'add';
    h = patch(ha,[x,fliplr(x)],[y+sy,fliplr(y-sy)],color);
    h.FaceAlpha = 0.3;
    plot(ha,x,y,'LineWidth',2,'Color',color);
end

