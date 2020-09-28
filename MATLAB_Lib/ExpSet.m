classdef ExpSet < handle
   
    properties(SetAccess=private)
        setData,
    end
    
    properties(Dependent)
        nExp,
        validField,
    end
    
    methods
        function obj = ExpSet()
            obj.setData = matList();
        end
        
        function addFile(obj)
            [fn,fp,index] = uigetfile('*.dldata');
            if index
                fpath = strcat(fp,fn);
                fid = fopen(fpath,'r');
                parseState = 0;
                strs = fgetl(fid);
                while(~isnumeric(strs))
                    if strs(1) == '#'
                        if parseState ~= 1
                            obj.setData.addOne(ExpInfo());
                        end
                        str = strsplit(strs(3:end),':');
                        obj.setData.data{obj.setData.len}.setInfo(str{1},str{2}(2:end));
                        parseState = 1;
                    else
                        parseState = 2;
                        str = strsplit(strs,',');
                        L = length(str);
                        for m = 1:2:L
                            obj.setData.data{obj.setData.len}.addData(str{m},str2double(str{m+1}));
                        end
                    end
                    strs = fgetl(fid);
                end
                fclose(fid);
            end
        end
        
        function delExps(obj,ids)
            obj.setData.delEles(ids);
        end
        
        function n = get.nExp(obj)
            n = obj.setData.len;
        end
        
        function v = get.validField(obj)
            if obj.nExp > 0
                v = obj.setData.data{1}.fields.toCell();
                for m = 2:obj.nExp
                    v = intersect(v,obj.setData.data{m}.fields.toCell());
                end
            else
                v = {};
            end
        end
        
        function plotFields(obj,id,f1,f2,ha)
            if ~exist('ha','var')
                ha = gca;
            end
            if length(id) == 1
                plot(ha,obj.setData.data{id}.getData(f1),...
                    obj.setData.data{id}.getData(f2),'LineWidth',1);
            else
                ha.NextPlot='add';
                for m = 1:length(id)
                    plot(ha,obj.setData.data{id(m)}.getData(f1),...
                        obj.setData.data{id(m)}.getData(f2),'LineWidth',1,...
                        'DisplayName',num2str(m));
                end
            end
            xlabel(f1);
            ylabel(f2);
        end
        
        function r = info(obj,index)
            r = obj.setData.data{index}.info;
        end
    end
end

