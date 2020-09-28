function saveFigSer( hf,fname )
    if ~exist('fname','var')
        [fn,fp] = uiputfile();
        I = strfind(fn,'.');
        fn = fn(1:(I(end)-1));
        fname = strcat(fp,fn);
    end
    saveas(hf,strcat(fname,'.fig'),'fig');
    saveas(hf,strcat(fname,'.eps'),'epsc');
    saveas(hf,strcat(fname,'.png'),'png');
end

