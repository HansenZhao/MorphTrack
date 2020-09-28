function [ histArray ] = cellularAuto( rules,len,steps,init,world,isShow )
    if ~exist('rules','var')
        rules = 110;
    end
    
    if ~exist('len','var')
        len = 21;
    end
    
    if ~exist('steps','var')
        steps = 100;
    end
    
    if ~exist('init','var')
        init = randi(2,[1,len])-1;
    elseif size(init,2)~= len
        error('Init condition size: %d is not equal to the length: %d\n',size(init,2),len);
    end
    
    if ~exist('world','var')
        world = 'zero';
    end
    
    if ~exist('isShow','var')
        isShow = 1;
    end
    
    histArray = [init;zeros(steps-1,len)];
    rules = uint8(rules);
    for m = 2:1:steps
        tmp = histArray(m-1,:);
        switch world
            case 'zero'
                tmp = [0,tmp,0];
            case 'one'
                tmp = [1,tmp,1];
            case 'extend'
                tmp = [tmp(1),tmp,tmp(end)];
            case 'wrap'
                tmp = [tmp(end),tmp,tmp(1)];
            otherwise
                tmp = [0,tmp,0];
        end
        mat = [tmp(1:(end-2));tmp(2:(end-1));tmp(3:end)];
        condition = sum(mat.*repmat([4;2;1],1,len))+1;
        res = bitget(rules,condition);
        histArray(m,:) = res;
    end
    
    if isShow
        figure;
        imagesc(histArray); colormap([1,1,1;0,0,0]);
    end
end

