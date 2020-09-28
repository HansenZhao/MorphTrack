classdef ExpInfo < handle
   
    properties(SetAccess=private)
        info,
        fields,
        data
    end
    
    methods
        function obj = ExpInfo()
            obj.info = struct();
            obj.fields = matList();
            obj.data = matList();
        end
        
        function setInfo(obj,infoName,info)
            if ~isnan(str2double(info))
                info = str2double(info);
            end
            obj.info = setfield(obj.info,infoName,info);
        end
        
        function addData(obj,fieldName,data)
            [b,I] = ismember(fieldName,obj.fields.toCell());
            if b
                obj.data.data{I}.addOne(data);
            else
                obj.fields.addOne(fieldName);
                obj.data.addOne(matList);
                [~,I] = ismember(fieldName,obj.fields.toCell());
                obj.data.data{I}.addOne(data);
            end
        end
        
        function r = getData(obj,fieldName)
            [b,I] = ismember(fieldName,obj.fields.toCell());
            if b
                r = obj.data.data{I}.toMat();
            else
                error('ExpInfo: cannot found filed %s',fieldName);
            end
        end
    end
end

