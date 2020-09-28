function [Fs,s,gam,hf1,hf2] = mDFA(vec,order,isModified, partition)
    %DFA: Kantelhardt JW, Koscielny-Bunde E, Rego HHA, Havlin S, Bunde A. 
    % Detecting long-range correlations with detrended fluctuation analysis. 
    % Physica A: Statistical Mechanics and its Applications 295, 441-454 (2001).
    if ~exist('isModified','var')
        isModified = 0;
    end
    if ~exist('partition','var')
        partition = 16;
    end
    if length(vec) < 1000
        return;
    end
    s = ((order+2):2:floor(length(vec)/4))';
    Fs = zeros(length(s),1);
    I = 1:length(s);
    for n = 1:1:ceil(length(s)/partition)
        index = I(max([1,(end-partition+1)]):end);
        if isModified
            parfor m = index
                Fs(m) = getModFs(vec,s(m),order);
                %fprintf(1,'s: %d, fs: %.3f\n',s(m),Fs(m));
            end
        else
            parfor m = index
                Fs(m) = getFs(vec,s(m),order,0);
                %fprintf(1,'s: %d, fs: %.3f\n',s(m),Fs(m));
            end
        end
        I(max([1,(end-partition+1)]):end) = [];
    end
    fo = fit(s,Fs,'power1');
    hf1 = figure('Position',[0,0,800,800]); scatter(s,Fs,'filled'); hold on; plot(s,fo(s));
    title(sprintf('DFA%d: F_{(s)}=%.3f*s^{%.3f}',order,fo.a,fo.b));
    hf2 = figure('Position',[0,0,800,800]); scatter(log(s),log(Fs./power(s,0.5)),'filled');
    title(num2str(abs(fo.b-0.5)));
    gam = fo.b;
end


function mFs = getModFs(vec,s,order)
    Fs = getFs(vec,s,order,0);
    L = length(vec); s_normalize = floor(L/20);
    shuffTime = 10;
    ks = zeros(shuffTime,1);
    ks_n = zeros(shuffTime,1);
    for m = 1:1:shuffTime
        I = randsample(1:L,L);
        ks(m) = getFs(vec(I),s,order,0);
        ks_n(m) = getFs(vec(I),s_normalize,order,0);
    end
    mFs = Fs * sqrt(mean(ks_n.^2)*s)/sqrt(mean(ks.^2)*s_normalize);
end


function Fs = getFs(vec,s,order,isDebug)
    vec = cumsum(vec);
    L = length(vec);
    Ns = floor(L/s);
    binIndex = [1:(Ns*s),L:-1:(L-Ns*s+1)];
    binIndex = reshape(binIndex,s,2*Ns); %partition in each column
    mat = vec(binIndex);
    modelName = sprintf('poly%d',order);
    if isDebug
        figure; subplot(211); hold on;
    end
    for m = 1:1:(2*Ns)
        Y = mat(:,m);
        fobject = fit((1:s)',Y,modelName);
        estY = fobject(1:s);
        if isDebug && m <= Ns
            plot(binIndex(:,m),mat(:,m),'b');
            plot(binIndex(:,m),estY,'r');
        end
        mat(:,m) = Y - estY;
    end
    if isDebug
        subplot(212); hold on;
        plot(1:(Ns*s),reshape(mat(:,1:Ns),Ns*s,1));
    end
    Fv = mean(mat.^2);
    Fs = sqrt(mean(Fv));
end

