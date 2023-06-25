function [HRVfeatures,HRVfeatures_names] = GetHRVfeatures(j,T,sample_fre,firstdpt,lastdpt,ECG_col,NIBP_col,PlotFreAnalaysis)
%HRVFEATURES Summary of this function goes here
%   Tor Finseth
%   3/12/20
%
% EXAMPLE INPUTS/OUTPUTS
%   j = epoch/bin interval
%   T = Table that has features/signals as columns, with
%   obervations/datapoints as rows. 
%   sample_fre = 125; %Hz
%   firstdpt = 30; The table starting index(row) for the bin
%   lastdpt = 60; The table last index(row) for the bin
%   ECG_col = 3; The column of the table that has the ECG raw signal
%   NIBP_col = 5; The column of the table that has the NIBP raw signal
%   PlotFreAnalaysis = 0 or 1; Do you want frequncy anlaysis?

%%    %%%%%%%%%%%% Phasic Response/second %%%%%%%%%%%%%%%%%%%%
    % calculate signals from EDAphasic input
%     eda_phasic=T.EDAphasic(firstdpt:lastdpt);
%     [pks,locs] = findpeaks(eda_phasic);
%        pks=1;
%     RatePhasic(j,1)=length(pks)/Epochsize;

%%    %%%%%%%%%%%% Respiration spectral entropy %%%%%%%%%%%%%%%%
    %Subband spectral entropy (Spectral entropy calculated from frequency band 0.05Hz to 0.35Hz)
%     if ismember('RSPXRSPECR', T.Properties.VariableNames)
%         SEResp(j,1) = pentropy(T.RSPXRSPECR(firstdpt:lastdpt),sample_fre,'Instantaneous',false,'FrequencyLimits',[0.05 0.35]);
%     else
%         SEResp(j,1)=1; %If this doesnt exist in the dataset, let the classfiyer delete it.
%     end

%%    %%%%%%%%%%%% Calculate - HRV from ECG %%%%%%%%%%%%%%%%%%%%
        % calculate signals from ECG input
        ecg_signal=T{firstdpt:lastdpt,ECG_col};
        NIBP_signal=T{firstdpt:lastdpt,NIBP_col};
        %Locate peaks
        
        [qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(ecg_signal,sample_fre,0); %1=plot,0=noplot
        
        if PlotFreAnalaysis==1
            % Power Spectral -ECG
            Fs=sample_fre;
            x=ecg_signal;
            M = length(x);
            xdft = fft(x);
            xdft = xdft(1:M/2+1);
            psdx = (1/(Fs*M)) * abs(xdft).^2;
            psdx(2:end-1) = 2*psdx(2:end-1);
            freq = 0:Fs/length(x):Fs/2;
            figure;
            plot(freq,10*log10(psdx))
            grid on
            title('ECG - Periodogram Using FFT')
            xlabel('Frequency (Hz)')
            ylabel('Power/Frequency (dB/Hz)')
            %NIBP
            x=NIBP_signal;
            M = length(x);
            xdft = fft(x);
            xdft = xdft(1:M/2+1);
            psdx = (1/(Fs*M)) * abs(xdft).^2;
            psdx(2:end-1) = 2*psdx(2:end-1);
            freq = 0:Fs/length(x):Fs/2;
            figure;
            plot(freq,10*log10(psdx))
            grid on
            title('NIBP - Periodogram Using FFT')
            xlabel('Frequency (Hz)')
            ylabel('Power/Frequency (dB/Hz)')
            % Mel-Spectrogram
            S = melSpectrogram(ecg_signal,sample_fre, ...
                       'WindowLength',size(ecg_signal,1),...
                       'OverlapLength',round(size(ecg_signal,1)/2), ...
                       'FFTLength',size(ecg_signal,1)*4, ...
                       'NumBands',256, ...
                       'FrequencyRange',[2,60]);
            [numBands,numFrames] = size(S);
            fprintf("Number of bandpass filters in filterbank: %d\n",numBands)
            fprintf("Number of frames in spectrogram: %d\n",numFrames)
            figure;
            melSpectrogram(ecg_signal,sample_fre, ...
                       'WindowLength',round(size(ecg_signal,1)/8),...
                       'OverlapLength',round(size(ecg_signal,1)/16), ...
                       'FFTLength',size(ecg_signal,1), ...
                       'NumBands',64, ...
                       'FrequencyRange',[10,60]);
            title('ECG Mel-Spectrogram')
            %NIBP
            figure;
            melSpectrogram(NIBP_signal,sample_fre, ...
                       'WindowLength',round(size(NIBP_signal,1)/8),...
                       'OverlapLength',round(size(NIBP_signal,1)/16), ...
                       'FFTLength',size(NIBP_signal,1), ...
                       'NumBands',64, ...
                       'FrequencyRange',[10,60]);
            title('NIBP Mel-Spectrogram')
            %Scalogram
            figure;
            cwt(ecg_signal,'amor',Fs)
            title('ECG Scalogram')
            colormap(pink(240))
            wt= cwt(ecg_signal,'amor',Fs); %get the wavelet transform
            %NIBP
            figure;
            cwt(NIBP_signal,'amor',Fs)
            title('NIBP Scalogram')
            colormap(pink(240))
            wt= cwt(NIBP_signal,'amor',Fs); %get the wavelet transform
            
            addpath('C:\Users\tfinseth\Documents\MATLAB\Add-Ons\Collections\ArrangePlotWindows');
            autoArrangeFigures(2,3,1);
            disp('Press space to continue, Ctrl+c to exit')
            pause  
        end
        
        %compute RR
        for num=1:1:length(qrs_i_raw)-1
            RR(num)=((qrs_i_raw(num+1)-qrs_i_raw(num))/100); %(seconds)
            RR(num)=RR(num)-0.2180; %to match this RR with what is produced by biopac (this calcualation RR is alittle high)
        end
        if (isempty(num) || num==1)
            RR=[];
        end
        RR = RR(:); %not enough resolution
        mean(RR);
        %RR = HRV.RRfilter(RR,0.15); %results in fft producing NaN values
        %mean(RR,'omitnan');
        %[avgHR_ep,meanRR_ep,rmssd_ep,nn50_ep,pNN50v,sd_RR_ep,sd_HR_ep,pse_ep,average_hrv_ep,hrv_ep]= getECGFeatures(ecg_signal,sample_fre);
        %addpath('C:\Users\tfinseth\Documents\MATLAB\Add-Ons\Collections\MarcusVollmer_HRV\MarcusVollmer-HRV-564b51f')
        avgHR= HRV.HR(RR,0); %HR=60/RR 
        rmssd= HRV.RMSSD(RR,0,0);
        pNN50= HRV.pNN50(RR,0,0);
        [epLF,epHF,eLFHFratio,eVLF,eLF,eHF] = HRV.fft_val_fun(RR,sample_fre);
        pLF= epLF; %normalized
        pHF= epHF; %normalized
        LFHFratio= eLFHFratio;
        VLF= eVLF;
        LF= eLF;
        HF= eHF;  
        
        SDSD=HRV.SDSD(RR,0,0);
        SDNN=HRV.SDNN(RR,0,0); %Compute standard deviation of NN intervals
        pNN20=HRV.pNNx(RR,0,20,0); % Probability of intervals greater x ms or smaller -x ms
        TRI=HRV.TRI(RR,0); % Compute Triangular index from the interval histogram
        TINN=HRV.TINN(RR,0); % Compute TINN, performing Triangular Interpolation
        [eSD1,eSD2,eSD1SD2ratio] = HRV.returnmap_val(RR,0,0); %Results of the Poincare plot (SD1,SD2,ratio)
        SD1= eSD1;
        SD2= eSD2;
        SD1SD2ratio= eSD1SD2ratio;
       
        HRVfeatures= {avgHR,rmssd,pNN50,pLF,pHF,LFHFratio,VLF,LF,HF,SDSD,SDNN,pNN20,TRI,TINN,SD1,SD2,SD1SD2ratio};
        HRVfeatures_names= {'avgHR','rmssd','pNN50','pLF','pHF','LFHFratio','VLF','LF','HF','SDSD','SDNN','pNN20','TRI','TINN','SD1','SD2','SD1SD2ratio'};   
    
end

