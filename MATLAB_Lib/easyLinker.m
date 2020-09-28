classdef easyLinker < handle
  
    properties
        refData,
        childData,
        colData,
        refComd,
        childComd,
        maskCell,
        colorCell,
        baseColor
    end
    
    properties (Dependent)
        dataSize,
        nMask
    end
    
    methods
        function obj = easyLinker(ref,child,refComd,childComd,col)
            if ~exist('refComd','var')
                refComd = 'scatter';
            end
            if ~exist('childComd','var')
                childComd = 'scatter';
            end
            if ~exist('col','var')
                col = 1:size(child,2);
            end
            obj.colData = col;
            obj.refData = ref;
            if size(child,1) ~= obj.dataSize
                error('row number of refData and childData should be consist,get %d,%d',...
                    obj.dataSize,size(child,1));
            end
            if size(child,2) ~= length(obj.colData)
                error('column number of childData and colData should be consist,get %d,%d',...
                    length(obj.colData),size(child,2));
            end
            obj.childData = child;
            obj.refComd = refComd;
            obj.childComd = childComd;
            obj.maskCell = {};
            obj.colorCell = {};
            obj.baseColor = 0.94*ones(1,3);
        end
        function r = get.dataSize(obj)
            r = size(obj.refData,1);
        end
        function r = get.nMask(obj)
            r = length(obj.maskCell);
        end
        function show(obj,marker,ha,includeMask)
            if ~exist('marker','var')
                marker = 1;
            end
            if ~exist('ha','var')
                hf=figure;
                ha = axes('Parent',hf);
            end
            if ~exist('includeMask','var')
                includeMask = 1:obj.nMask;
            end
            if marker == 1
                data = obj.refData;
                comd = obj.refComd;
                obj.plotWithComd(ha,comd,data,obj.baseColor);
            elseif marker == 2
                data = obj.childData;
                comd = obj.childComd;
            else
                error('unsolved marker, should be 1 or 2, get %d',marker);
            end
            if includeMask
                ha.NextPlot = 'add';
                L = length(includeMask);
                for m = 1:L
                    index = includeMask(m);
                    obj.plotWithComd(ha,comd,data(obj.maskCell{index},:),obj.colorCell{index});
                end
            end
        end
        function dataSwitch(obj)
            tmp = obj.refData;
            obj.refData = obj.childData;
            obj.childData = tmp;
        end
        function maskRegion(obj)
            hf = figure;
            obj.show(1,gca,1:obj.nMask)          
            a = impoly;
            pXpY = a.getPosition();
            isIn = inpolygon(obj.refData(:,1),obj.refData(:,2),pXpY(:,1),pXpY(:,2));
            close(hf);
            obj.maskCell{end+1} = isIn;
            obj.colorCell{end+1} = rand(1,3);
            obj.show(1);
            obj.show(2);
        end
        function setColor(obj,n)
            if n > obj.nMask || n < 1
                error('invalid index, should be 1 ~ %d, get %d',obj.nMask,n);
            end
            obj.colorCell{n} = uisetcolor();
        end
        function maskUnion(obj,n,newN)
            I = ismember(n,1:obj.nMask);
            if all(I)
                L = length(n);
                tmp = obj.maskCell{n(1)};
                for m = 2:L
                    tmp = or(tmp,obj.maskCell{n(m)});
                end                
                if ~exist('newN','var')
                    [minN,index] = min(n);
                    obj.maskCell{minN} = tmp;
                    if L > 1
                        n(index) = [];
                        obj.maskCell(n) = [];
                    end
                elseif strcmp(newN,'new')
                    obj.maskCell{end+1} = tmp;
                elseif ismember(newN,n)
                    obj.maskCell{newN} = tmp;
                    if L > 1
                        n(n==newN) = [];
                        obj.maskCell(n) = [];
                    end
                else
                    error('unsolve newN param, should be ''new'' or a member in ''n'' param');
                end             
            else
                index = find(I==0);
                error('unexpect index, should be 1~%d, get: %d',obj.nMask,n(index(1)))
            end
        end
        function newObj = birthWithMask(obj,n)
            if n > obj.nMask || n < 1
                error('invalid index, should be 1 ~ %d, get %d',obj.nMask,n);
            end
            newObj = easyLinker(obj.refData(obj.maskCell{n},:),...
                                obj.childData(obj.maskCell{n},:),...
                                obj.refComd,obj.childComd);
        end
        function I = toIndex(obj)
            I = zeros(obj.dataSize,1);
            for m = 1:obj.nMask
                I(obj.maskCell{m}) = m;
            end
        end
    end
    
    methods(Access = private)
        function plotWithComd(obj,ha,comd,X,c)
            if ~exist('c','var')
                c = zeros(1,3);
            end
            [nr,nc] = size(X);
            if nc < 2
                X = [(1:nr)',X];
            end
            switch comd
                case 'plot'
                    plot(ha,obj.colData,mean(X,1),'LineWidth',1,'Color',c);
                case 'scatter'
                    scatter(ha,X(:,1),X(:,2),10,c,'filled');
                case 'bar'
                    bar(ha,obj.colData,mean(X,1),'FaceColor',c);
                otherwise
                    error('unsolved comd, should be ''plot'',''scatter'' or ''bar'', get: %s',comd);
            end
        end
    end 
end

