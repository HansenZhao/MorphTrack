function imMat2txt( M,syb,fileName )
    %UNTITLED Summary of this function goes here
    % imMat2txt( M,syb,fileName )
    [nr,nc] = size(M);
    if ~exist('syb','var')
        syb = '%.3f';
    end
    if ~exist('fileName','var')
        [fn,fp] = uiputfile('*.txt');
        fileName = strcat(fp,fn);
    end
    fid = fopen(fileName,'w');
    for m = 1:1:nr
        for n = 1:1:(nc-1)
            fprintf(fid,sprintf('%s ',syb),M(m,n));
        end
        fprintf(fid,sprintf('%s \n',syb),M(m,nc));
    end
    fclose(fid);
end

