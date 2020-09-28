classdef matList < handle
    
    properties(SetAccess=private)
        capacity;
        len;
        data;
    end
    
    methods
        function obj = matList(initCap)
            if ~exist('initCap','var')
                initCap = 100;
            end
            obj.capacity = initCap;
            obj.data = cell(obj.capacity,1);
            obj.len = 0;
        end
        
        function addOne(obj,mat)
            if obj.len == obj.capacity
                obj.expandCap();
            end
            obj.len = obj.len + 1;
            obj.data{obj.len} = mat;
        end
        
        function b = isExist(obj,mat)
            b = any(cellfun(@(x)isequal(x,mat),obj.data(1:obj.len)));
        end
        
        function [b,m] = isExist2(obj,mat,acc)
            b = 0;
            if ~exist('acc','var')
                acc = 1e-6;
            end
            for m = 1:obj.len
                b = all(abs(obj.data{m}-mat)<acc);
                if b
                    return;
                end
            end
            m = -1;
        end
        
        function replaceEle(obj,index,newEle)
            obj.data{index} = newEle;
        end
        
        function saveAsTable(obj,fileName,varNames)
            L = length(obj.data{1}(:));
            if L ~= length(strsplit(varNames,','))
                error('matList: varNames length %d is inconsist to the data length %d!',...
                    length(varNames),L);
            end
            tmp = cellfun(@(x)length(x(:)),obj.data(1:obj.len));
            if any(tmp~=L)
                error('matList: there exists inconsist data length among data');
            end
            t = cell2mat(cellfun(@(x)x(:)',obj.data(1:obj.len),'UniformOutput',0));
            HScsvwrite(fileName,t,[],varNames);
        end
        
        function saveAsMAT(obj,fileName)
            data = obj.data(1:obj.len);
            save(fileName,'data');
        end
        
        function r = toMat(obj)
            r = cell2mat(obj.data(1:obj.len));
        end
        
        function r = toCell(obj)
            r = obj.data(1:obj.len);
        end
        
        function delEles(obj,indices)
            obj.data(indices) = [];
            newL = 0;
            for m = 1:obj.len
                if ~isempty(obj.data{m})
                    newL = newL + 1;
                    obj.data{newL} = obj.data{m};
                end
            end
            obj.len = newL;
        end
    end
    
    methods(Access=private)
        function expandCap(obj)
            tmp = cell(2*obj.capacity,1);
            tmp(1:obj.len,:) = obj.data;
            obj.data = tmp;
            obj.capacity = 2 * obj.capacity;
        end
    end
    
end

