function [newX,newY,newStdU,newStdD] = stdline(x,vec,tags,ha)
    x = x(:);
    isEqual = length(vec)==length(tags);
    L = length(x);
    [newX,newY,newStdU,newStdD] = deal(zeros(2*L,1));
    starter = 1; tmp = tags(1); pointer = 2; newX(1)= 1;
    for m = 2:1:L
        if tags(m) ~= tmp
            newX(pointer:(pointer+1)) = [m-1;m];
            if isEqual
                newY((pointer-1):pointer) = mean(vec(starter:(m-1)));
                newStdU((pointer-1):pointer) = mean(vec(starter:(m-1)))+std(vec(starter:(m-1)));
                newStdD((pointer-1):pointer) = mean(vec(starter:(m-1)))-std(vec(starter:(m-1)));
            else
                newY((pointer-1):pointer) = vec(pointer/2);
            end
            starter = m;
            pointer = pointer + 2;
            tmp = tags(m);
        elseif m==L
            newX(pointer) = m;
            if isEqual
                newY((pointer-1):pointer) = mean(vec(starter:m));
                newStdU((pointer-1):pointer) = mean(vec(starter:m))+std(vec(starter:m));
                newStdD((pointer-1):pointer) = mean(vec(starter:m))-std(vec(starter:m));
            else
                newY((pointer-1):pointer) = vec(pointer/2);
            end
        end
    end
    newX((pointer+1):end) = [];
    newY((pointer+1):end) =[];
    newStdU((pointer+1):end) =[];
    newStdD((pointer+1):end) =[];
    
    if exist('ha','var')
        rectX = [x(newX);flipud(x(newX))];
        rectY = [newStdU;flipud(newStdD)];
        patch(ha,rectX,rectY,[0.8,0.8,0.8],'EdgeColor','none');
        hold on;
        plot(ha,x(newX),newY,'LineWidth',2);
        hold off;
    end
end

