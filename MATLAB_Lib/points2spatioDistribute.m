function [X,Y,Z] = points2spatioDistribute(points,resolution,padding,hFunc,varargin)
    % points: X|Y|value
    % vecStatistic = hFunc(vec)
    % example: normal distribution
    % [X,Y,Z] = points2spatioDistribute([randn(10000,2),ones(10000,1)],1,0,@(x)sum(x));
    % contour(X,Y,Z)
    if nargin == 4
        [minX,maxX] = deal(min(points(:,1)-padding),max(points(:,1))+padding);
        [minY,maxY] = deal(min(points(:,2)-padding),max(points(:,2))+padding);
    else
        
        [minX,maxX,minY,maxY] = deal(varargin{:});
    end
    
    resX = (maxX-minX)/resolution;
    resY = (maxY-minY)/resolution;
    
    [X,Y] = meshgrid(minX:resX:maxX,minY:resY:maxY);
    
    [nr,nc] = size(X);
    Z = zeros(nr,nc);
    
    for m = 1:1:nr
        row_y = Y(m,1);
        mask_y = abs(points(:,2)-row_y) <= (resY/2);
        if sum(mask_y)==0
            continue;
        else
            selected_points = points(mask_y,:);
            for n = 1:1:nc
                col_x = X(1,n);
                mask_x = abs(selected_points(:,1)-col_x) <= (resX/2);
                if sum(mask_x) == 0
                    continue;
                else
                    Z(m,n) = hFunc(selected_points(mask_x,3));
                end
            end
        end
    end   
end

