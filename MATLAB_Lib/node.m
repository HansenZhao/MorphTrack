classdef node < handle
    
    properties(SetAccess = private)
        data;
        prevNode;
        nextNode;
    end
    
    methods
        function obj = node(data,prev)
            if ~exist('prev','var')
                obj.prevNode = [];
            else
                obj.prevNode = prev;
            end
            obj.nextNode = [];
            obj.data = data;
        end
        function setData(obj,data)
            obj.data = data;
        end
        function linkPrev(obj,h)
            obj.prevNode = h;
        end
        function linkNext(obj,h)
            obj.nextNode = h;
        end
    end 
end

