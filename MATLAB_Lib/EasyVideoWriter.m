classdef EasyVideoWriter<handle
    
    properties
        aviObj;
        hFig;
    end
    
    methods
        function obj = EasyVideoWriter(hFigure)
            [fn,fp,~] = uiputfile('*.avi');
            obj.hFig = hFigure;
            Ans = inputdlg({'frame rate:','quality:'},'AVI setting',1,{'30','100'});
            obj.aviObj = VideoWriter(strcat(fp,fn));
            obj.aviObj.FrameRate = str2double(Ans{1});
            obj.aviObj.Quality = str2double(Ans{2});
            open(obj.aviObj);
        end
        
        function ticVideo(obj)
            writeVideo(obj.aviObj,getframe(obj.hFig));
        end
        
        function endVideo(obj)
            close(obj.aviObj);
            warndlg('Video Done!');
        end
    end
    
end

