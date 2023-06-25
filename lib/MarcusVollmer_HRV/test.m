 clc;
 clear;
 close all;
 fignum=1;
       HR = repmat(60,30,1);
      RR = [normrnd(1,.005,size(HR,1)/3,3)-repmat([-.045 .05 .025],...
      size(HR,1)/3,1)]';
      RR = RR(:);

      % Downsampling and plot the RR tachogram:
      Fs = 128;
      RR = round(RR*Fs)/Fs;
      Ann = cumsum(RR);
      figure(fignum)
      fignum=fignum+1;
      plot(Ann,RR)
      

      % Corresponding relative RR intervals:
      rr = HRV.rrx(RR);
      figure(fignum)
fignum=fignum+1;
      plot(Ann,rr)
      figure(fignum)
      fignum=fignum+1;
      plot(rr(1:end-1),rr(2:end),'Marker','o',...
      'MarkerFaceColor',1*[1 1 1],'MarkerEdgeColor',0*[1 1 1],...
      'MarkerSize',10,'Color',0.5*[1 1 1])

      % Compute certain HRV measures for continuously for 60 successive
      % RR intervals: 
      rmssd = HRV.RMSSD(RR,60);
      rrhrv = HRV.rrHRV(RR,60);
  figure(fignum)
fignum=fignum+1;
      plotyy(Ann,rmssd,Ann,rrhrv)
      
      %Ann = rdann('mitdb/100','atr');
      
      Fs = 250;
      Ann = Ann/Fs;
      RR = [NaN; diff(Ann)];
        figure(fignum)
fignum=fignum+1;
      % The RR tachogram shows obvious artifacts:
      plot(Ann,RR)
figure(fignum)
fignum=fignum+1;
      % Filter from artifacts and plot the average heart rate:
      RR = HRV.RRfilter(RR,0.15);
      plot(Ann,RR)
 figure(fignum)
fignum=fignum+1;
      % Plot the average heart rate:
      plot(Ann,HRV.HR(RR,60))
 figure(fignum)
fignum=fignum+1;
      % Corresponding relative RR intervals:
      rr = HRV.rrx(RR);
      plot(Ann,rr)
figure(fignum)
fignum=fignum+1;
      % Compute certain HRV measures for continuously for 60 successive
      % RR intervals: 
      rmssd = HRV.RMSSD(RR,60);
      rrhrv = HRV.rrHRV(RR,60);
      plotyy(Ann,rmssd,Ann,rrhrv)