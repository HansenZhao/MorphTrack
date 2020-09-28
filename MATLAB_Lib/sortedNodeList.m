classdef sortedNodeList < handle
    
    properties(SetAccess = private)
        compareFunc;        
        header;   
        len;
        tailer;
    end
    
    properties(Access = private)
        cacheData;
    end
    
    methods
        function obj = sortedNodeList(data,compareFunc)
            if ~exist('compareFunc','var')
                obj.compareFunc = @(a,b)a>=b;
            else
                obj.compareFunc = compareFunc;
            end
            if ~exist('data','var')
                obj.header = [];
                obj.len = 0;
                obj.tailer = [];
            else
                obj.insertAt(data,1);
            end          
        end
        
        function I = insertEle(obj,data)
            I = obj.findPos(data);
            obj.insertAt(data,I);
        end
        
        function res = contains(obj,data)
            if obj.len == 0
                res = 0;
                return;
            end
            acc = 1e-6;
            [~,a,b] = obj.findPos(data);
            res = all(abs(obj.cacheData{a}.data-data)<acc) || ...
                (b <= obj.len && all(abs(obj.cacheData{b}.data-data)<acc));
        end
        
        function e = getEle(obj,index)
            if index > 0 && index <= obj.len
                e = obj.cacheData{index}.data;
            else
                error('SORTEDNODELIST: invalid index: %d',index);
            end
        end
        
        function c = toCell(obj)
            c = cell(obj.len,1);
            for m = 1:obj.len
                c{m} = obj.cacheData{m}.data;
            end
        end
        
    end
    
    methods(Access = private)
        function updateCache(obj)
            obj.cacheData = cell(obj.len,1);
            tmp = obj.header;
            for m = 1:obj.len
                obj.cacheData{m} = tmp;
                tmp = tmp.nextNode;
            end
        end
        function insertAt(obj,data,index)
            if obj.len == 0
                obj.header = node(data);
                obj.tailer = obj.header;
            elseif index == (obj.len + 1)
                tmpNode = node(data,obj.tailer);
                obj.tailer.linkNext(tmpNode);
                obj.tailer = tmpNode;
            elseif index == 1
                tmpNode = node(data);
                tmpNode.linkNext(obj.header);
                obj.header.linkPrev(tmpNode);
                obj.header = tmpNode;
            else
                tmpNode = node(data,obj.cacheData{index-1});
                obj.cacheData{index}.linkPrev(tmpNode);
                obj.cacheData{index-1}.linkNext(tmpNode);
                tmpNode.linkNext(obj.cacheData{index});
            end
            obj.len = obj.len + 1;
            obj.updateCache();
        end
        function [I,beginner,ender] = findPos(obj,data)
            if obj.len == 0
                I = 1;
                beginner = [];
                ender = [];
            else
                beginner = 1;
                ender = obj.len+1;
                while(ender-beginner > 1)
                    tmpIndex = ceil((beginner+ender)/2);
                    if obj.compareFunc(data,obj.cacheData{tmpIndex}.data)
                        beginner = tmpIndex;
                    else
                        ender = tmpIndex;
                    end
                end
                if obj.compareFunc(data,obj.cacheData{beginner}.data)
                    I = ender;
                else
                    I = beginner;
                end
            end
        end
    end
    
end

