function [ comps ] = graphCompAna( adjMat )
    I = sum(adjMat,2);
    searchFrom = find(I>0,1);
    comps = matList();
    auxMat = adjMat;
    while(~isempty(searchFrom))
        comp = searchFrom;
        newComp = find(adjMat(searchFrom,:)>0);
        while(~all(ismember(newComp,comp)))
            comp = newComp;
            newComp = find(sum(adjMat(comp,:))>0);
        end
        comps.addOne(comp);
        auxMat(newComp,:) = 0;
        I = sum(auxMat,2);
        searchFrom = find(I>0,1);
    end    
end

