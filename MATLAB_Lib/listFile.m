function [fileName,filePath] = listFile(spec,path)
    fileName = {}; filePath = {};
    if nargin==1
        path = uigetdir();
    end
    files = ls(path);
    L = size(files,1);
    if L < 3
        return;
    end
    files = mat2cell(files,ones(1,L));
    index = strfind(spec,'.');
    matchName = spec(1:(index(end)-1));
    if strcmp(matchName,'*')
        matchName = [];
    end
    matchFormat = spec((index(end)+1:end));
%     fileType = strsplit(spec,'.');
%     if strcmp(fileType{1},'*')
%         matchName = [];
%     else
%         matchName = fileType{1};
%     end
%     fileType = fileType{2};
%     if strcmp(fileType,'*')
%         matchFormat = [];
%     else
%         matchFormat = fileType;
%     end
    for m = 1:1:L
        if files{m}(1) == '.'
            continue;
        end
        if isdir(strcat(path,'\',files{m}))
            [fn,fp] = listFile(spec,strcat(path,'\',files{m}));
            len = size(fn,1);
            if len > 0
                fileName = [fileName,fn{:}];
                filePath = [filePath,fp{:}];
            end
        else
            try
                tmp = files{m};
                tmp = strtrim(tmp);
                I = strfind(tmp,'.');
                fName = tmp(1:(I(end)-1));
                fFormat = tmp((I(end)+1):end);
%                 tmp = strsplit(tmp,'.');
%                 if length(tmp) == 2
%                     fName = tmp{1};
%                     fFormat = tmp{2};
%                 elseif length(tmp) > 2
%                     fFormat = tmp{end};
%                     fName = strcat(tmp{1:(end-1)});
%                 end
                if isempty(matchFormat) || strcmp(matchFormat,fFormat)
                    %if strcmp(name,spec)
                    if isempty(matchName) || regexp(fName,matchName)
                        fileName{end+1} = files{m};
                        filePath{end+1} = strcat(path,'\');
                    end
                end
            catch
            end
        end
    end
end

