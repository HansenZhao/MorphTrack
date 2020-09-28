function [p,freq] = easyFFT(t,signal,isShow)
    if nargin == 2
        isShow = 1;
    end
    t = t - t(1);
    samplePeriod = t(2)-t(1);
    sampleFreq = 1/samplePeriod;
    L = length(t);
    if mod(L,2) == 1
        t = t(2:end);
        signal = signal(2:end);
        L = L-1;
    end
    
    Y = fft(signal);
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    p = P1;
    
    freq = sampleFreq*(0:(L/2))/L;
    
    if isShow
        figure;
        plot(subplot(2,1,1),t,signal,'DisplayName','signal');
        xlabel('s'); ylabel('signal'); box on; legend show;
        
        plot(subplot(2,1,2),freq,p,'DisplayName','fft(signal)');
        xlabel('(Hz)'); box on; legend show;
    end  
end

