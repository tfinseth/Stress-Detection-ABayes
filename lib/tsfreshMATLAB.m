function [features,features_names] = tsfreshMATLAB(y,fs)
% TSFRESHMATLAB	Compute a feature vector from an input time series

%   TSFRESHMATLAB takes in a signal an calculates a bunch of different
%   features that each have a single output. Identifying and interpreting discriminating features between different classes of time series
%   For Tsfresh equations see https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html
%   For Tsfresh parameter values see https://tsfresh.readthedocs.io/en/v0.5.0/_modules/tsfresh/feature_extraction/settings.html
%   For feature processing times see Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time series feature extraction on basis of scalable hypothesis tests (tsfresh–a python package). Neurocomputing, 307, 72-77.

% NOTE: Does not include bool features, nor other features that take longer than 10^-2 sec to run (cf. Christ et al., 2018)

 %  FEATURES
    x=(1:length(y)).'; 
    
    %% Abs Energy
    abs_energy=dot(y,y);
    %% Absolue sum of changes
    abs_sum_changes=sum(abs(diff(y)));
    %% Agg autocorrelation
    %% Agg linear trend   ------- OMMITED DUE TO TIME
    %% AR_Coeff -----------%OMITTED, takes way too long to compute
%     mdl = arima(1,0,0); % 2 the lag order
%     ARCoeff = estimate(mdl,y,'Display','off').AR{1}; 
    %% Augmented Dickey_Fuller
    [~,adfpValue] = adftest(y);
    %% Autocorelation (mean of Lag 1)
    acf = autocorr(y);
    Agg_AutoC=acf(2);
    %% Binned entropy
    max_bins=10;
    binned_entropy=getEntropy(y,max_bins);
    %% C3 - measure of non linearity in the time series
    %     [1] Schreiber, T. and Schmitz, A. (1997).
    %     |  Discrimination power of measures for nonlinearity in a time series
    %     |  PHYSICAL REVIEW E, VOLUME 55, NUMBER 5
    lag=1;
    c2=(circshift(y, 2.*-lag).*circshift(y,-lag).*x);
    C3=mean(c2(1:(length(y)-2*lag)));
    %% Change Quantiles ------- OMMITED DUE TO TIME
    %% Cid_ce --- estimate for a time series complexity [1]
    %     Batista, Gustavo EAPA, et al (2014).
    %     |  CID: an efficient complexity-invariant distance for time series.
    %     |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.
    delta = diff(y);
    cid_ce=sqrt(dot(delta, delta));
    %% Count above Mean
    GTmean=sum(y > mean(y));
    %% Count below Mean
    LTmean=sum(y < mean(y));
    %% ctw coefficeints
%     CWTcoeffs = cwt(y,1:128,'mexh'); %check 
    %% energy_ratio_by_chunks
%     full_series_energy=sum(y^2);
    %% fft aggregated spectral centroid (mean)
    ffta = fft(y);
    rfft = ffta(1:(floor(length(ffta)/2)+1));
    fft_abs = abs(rfft);
    fft_agg_var=moment(fft_abs,2); %first moment (centriod is always zero for any dataset), 2nd moment is variance
     %% FFT Coeficient - equivalent to np.fft.rfft(a) in python
    ffta = fft(y);
    rfft = ffta(1:(floor(length(ffta)/2)+1));
%     rfft_img=imag(rfft); %just the imaginary component
%     rfft_img=max(abs(rfft_img(2:end,:)));
    rfft_real=real(rfft); %just the real component
    rfft_real=max(abs(rfft_real(2:end,:)));
   %% first_location of maximum
   %Returns the relative first location of the maximum value of y. The position is calculated relatively to the length of y.
    [~, argmax] = max(y);
    first_loc_max=argmax(1)/length(y); 
  %% first location of minimum
    [~, argmin] = min(y);
    first_loc_min=argmin(1)/length(y);
   %% index mass quatile
   %I dont think this works for my data, the cumsum(and cdf) come out as
   %linear, meaning the index of the mass will be equal intervals along the
   %line.
%    quant=[.1,.2,.3,.4,.5,.6,.7,.8,.9];
%     abs_y = abs(y);
%     s = sum(abs_y);
% 
%     if (s == 0)
%         % all values in x are zero or it has length 0
%         index_mass_q1=NaN;
%         index_mass_q2=NaN;
%         index_mass_q3=NaN;
%         index_mass_q4=NaN;
%         index_mass_q5=NaN;
%         index_mass_q6=NaN;
%         index_mass_q7=NaN;
%         index_mass_q8=NaN;
%         index_mass_q9=NaN;
%      else
%         % at least one value is not zero
%         mass_centralized = cumsum(abs_y) / s; %normlized cdf
%         [~, max_ind]=max(mass_centralized >= quant(1:9)); %find the index for the qunatile (e.g., index of top 20 %)
%         index_mass_q1=y(max_ind(1)) / length(y);
%         index_mass_q2=y(max_ind(2)) / length(y);
%         index_mass_q3=y(max_ind(3))) / length(y);
%         index_mass_q4=y(max_ind(4))) / length(y);
%         index_mass_q5=(y(max_ind(5))) / length(y);
%         index_mass_q6=(y(max_ind(6))) / length(y);
%         index_mass_q7=(y(max_ind(7))) / length(y);
%         index_mass_q8=y(max_ind(8))  / length(y);
%         index_mass_q9=(y(max_ind(9))) / length(y);
%      end
   %% kurtosis
   kurt=kurtosis(y);
   %% last location of maximum
   [~, argmax] = max(y);
    last_loc_max=argmax(end)/length(y);
   %% last location of minimum
    [~, argmin] = min(y);
    last_loc_min=argmin(end)/length(y);
   %% linear trend
    %Slope
    [p]=polyfit(x,y,1); %p(1)=slope, p(2)=intercept
    linear_slope= p(1);
   %% longest strike above mean
   runs=get_length_seq(y>mean(y));
   if size(runs)>0
        longest_strike_above= max(runs);
   else
       longest_strike_above=0;
   end     
       
   %% longest strike below mean
      runs=get_length_seq(y<mean(y));
   if size(runs)>0
        longest_strike_below= max(runs);
   else
       longest_strike_below=0;
   end   
   %% max langevin fixed point ------- OMMITED DUE TO TIME
    %% Maximum
%     epoch_max(j,i)= max(T{firstdpt:lastdpt,i},[],1); %Too correlated
    %% Mean
    M= mean(y,1);
    %% mean abs change
    mean_abs_change= mean(abs(diff(y)));
    %% mean change
    if length(y) > 1 
        mean_change=(y(end) - y(1)) / (length(y) - 1); 
    else
        mean_change=NaN;
    end
    %% mean second derivative central
    if length(x) > 2
        mean_2nd=(y(end) - y(end-1) - y(2) + y(1)) / (2 * (length(y) - 2)); 
    else
        mean_2nd= NaN;
    end
    %% median
    med=median(y);
    %% Minmum
%     epoch_min(j,i)= min(T{firstdpt:lastdpt,i},[],1);%Too correlated
    %% number crossings mean
    positive = y > mean(y);
    change=logical(abs(diff(positive)));
    [counts, ~] = hist(change,unique(change));
    if isempty(counts)
        num_cross_mean=0;
    else
        num_cross_mean= counts(2);
    end
    %% number cwt peaks  ------- OMMITED DUE TO TIME
    %% number peaks  ------- OMMITED DUE TO TIME
    %% Partial Autocorelations (Lag 1:5)
    warning('off','all'); %supress the warnings from this function
    pacf = parcorr(y);
    warning('on','all');
    Agg_PAutoC=mean(pacf(2:6));
    %% percent of reoccuring datapoints to all datapoints
    if isempty(y)
        per_reocurr_dtp=NaN;
    else
        try
        [counts, ~] = hist(y,unique(y));
        catch
            counts=0;
        end
        if size(counts,2)== 0
           per_reocurr_dtp=0;
        else
            per_reocurr_dtp=sum(counts > 1) / size(counts,2);
        end
    end
    %% percent of reoccuring values to all values
    if isempty(y)
        per_reocurr_val=NaN;
    else
        value_counts  = histc(y,unique(y));
        reoccuring_values = sum(value_counts(value_counts > 1));
        if isnan(reoccuring_values)
            per_reocurr_val=0;
        else
            per_reocurr_val=reoccuring_values / length(y);
        end
    end
    %% quantile  ------- OMMITED DUE TO TIME
    %% range count
%     minv=-1;
%     maxv=1;
%     range_count=sum((y >= minv) & (y < maxv));
    %% ratio_beyond_r_sigma
    %% ratio value number to time series length
    if isempty(y)
        ratio_val_totime_series= NaN;
    else
        ratio_val_totime_series=size(unique(y)) / size(y);
    end
    %% sample entropy ------- OMMITED DUE TO TIME
    %% skewness
    skew=skewness(y);
    %% spkt_welch_density
    spkt_coeff=[2,5,8];
    [pxx,f] = pwelch(y,fs);
    sptkt_power=pxx(spkt_coeff);
    psd_welchp_c1=sptkt_power(1);
    psd_welchp_c2=sptkt_power(2);
    psd_welchp_c3=sptkt_power(3);
%     psd_welchp=max(pxx);
%     psd_welchf=f(pxx==psd_welchp);
%     if size(psd_welchf,1)>1
%         if psd_welchp==0
%             psd_welchf=0;
%         end
%     end

%plot
% plot(f,10*log10(pxx))
% xlabel('Frequency (Hz)')
% ylabel('PSD (dB/Hz)')
    %% Standard deviation
    stddev= std(y,1);
    %% sum of reoccuring data points
    try
     [counts, unique_y] = hist(y,unique(y));
    catch
      counts=0;
      unique_y=0; 
    end
    counts(counts < 2) = 0;
    sum_reoccuring_dpt=sum(dot(counts, unique_y));
    %% sum of reoccuring values
    try
     [counts, unique_y] = hist(y,unique(y));
    catch
      counts=0;
      unique_y=0;  
    end
     
    counts(counts < 2) = 0;
    counts(counts > 1) = 1;
    sum_reoccuring_val=sum(dot(counts, unique_y));
    %% sum values
    sum_val=sum(y);
    %% symmetry looking -- Bool
    %% time reversal asymmetry statistic
    lag=1;
    n = length(y);
    if 2 * lag >= n
        time_reversal=0;
    else
        one_lag = circshift(y, -lag); 
        two_lag = circshift(y, 2 * -lag);
        time_diff=(two_lag .* two_lag .* one_lag - one_lag .* y .* y);
       time_reversal=mean(time_diff(1:(n-2*lag)));
    end
    %% value count
    %% variance
    variance=var(y);
     %% Sum of Squares
    SS= sumsqr(y);
    %% Root Mean Square
    %% epoch_RMS(j,i)= rms(T{firstdpt:lastdpt,i}); %Too correlated to the Mean
        %Range
    %% R= range(y); %Too Correlated
    %% Calculate - peak2peak-Maximum-to-minimum difference
    %% epoch_P2P(j,i)= peak2peak(T{firstdpt:lastdpt,i});   
    %% Calculate - Standard Deviation of the Standard Deviation Vector, or Standard Deviation of Succesvive Differences
    SDSD= std(diff(y,1));
    %% Calculate - Cross Correlation, mean
    CC= mean(xcorr(diff(y,1)));

    %%%%%Do not have welch power
    features={abs_energy,abs_sum_changes,adfpValue,Agg_AutoC,binned_entropy,C3,cid_ce, GTmean,LTmean,fft_agg_var,...
           rfft_real,first_loc_max,first_loc_min,kurt,last_loc_max,last_loc_min,linear_slope,longest_strike_above,longest_strike_below,M,...
           mean_abs_change,mean_change,mean_2nd,med, num_cross_mean,Agg_PAutoC,per_reocurr_dtp,per_reocurr_val,...
           ratio_val_totime_series,skew,stddev,sum_reoccuring_dpt,sum_reoccuring_val,sum_val,time_reversal,variance,SS,SDSD,CC};
    features_names= {'abs_energy','abs_sum_changes','adfpValue','Agg_AutoC','binned_entropy','c3','cid_ce', 'GTmean','LTmean','fft_agg_var',...
          'rfft_real','first_loc_max','first_loc_min','kurt','last_loc_max','last_loc_min','linear_slope','longest_strike_above','longest_strike_below','mean',...
           'mean_abs_change','mean_change','mean_2nd','median','num_cross_mean','Agg_PAutoC','per_reocurr_dtp','per_reocurr_val',...
           'ratio_val_totime_series','skew','stddev','sum_reoccuring_dpt','sum_reoccuring_val','sum_val','time_reversal','variance','SS','SDSD','CC'};

    %%%%Have welch power
%      features={abs_energy,abs_sum_changes,adfpValue,Agg_AutoC,binned_entropy,C3,cid_ce, GTmean,LTmean,fft_agg_var,...
%            rfft_real,first_loc_max,first_loc_min,kurt,last_loc_max,last_loc_min,linear_slope,longest_strike_above,longest_strike_below,M,...
%            mean_abs_change,mean_change,mean_2nd,med, num_cross_mean,Agg_PAutoC,per_reocurr_dtp,per_reocurr_val,...
%            ratio_val_totime_series,skew,psd_welchp_c1,psd_welchp_c2,psd_welchp_c3,stddev,sum_reoccuring_dpt,sum_reoccuring_val,sum_val,time_reversal,variance,SS,SDSD,CC};
%     features_names= {'abs_energy','abs_sum_changes','adfpValue','Agg_AutoC','binned_entropy','c3','cid_ce', 'GTmean','LTmean','fft_agg_var',...
%           'rfft_real','first_loc_max','first_loc_min','kurt','last_loc_max','last_loc_min','linear_slope','longest_strike_above','longest_strike_below','mean',...
%            'mean_abs_change','mean_change','mean_2nd','median','num_cross_mean','Agg_PAutoC','per_reocurr_dtp','per_reocurr_val',...
%            'ratio_val_totime_series','skew','psd_welchp_c1','psd_welchp_c2','psd_welchp_c3','stddev','sum_reoccuring_dpt','sum_reoccuring_val','sum_val','time_reversal','variance','SS','SDSD','CC'};

%     features= {M,stddev,slope,SS,SDSD,CC,Agg_AutoC,Agg_PAutoC,adfpValue,cid_ce,C3,GTmean,LTmean,rfft_img,rfft_real,psd_welchp,psd_welchf};
%     features_names= {'Mean','Std','slope','SS','SDSD','CC',...
%         'Agg_AutoC','Agg_PAutoC','adfp','CidCe','C3','GTmean','LTmean','rfft_img','rfft_real','psd_welchp','psd_welchf'};   
    
    %% Catch22 features instead --- not as good of accuracy as just using the other features
    [catfeat,catfeat_names]=catch22_all(y);
    features=[features,catfeat];
    features_names=[features_names,catfeat_names];

end


%%
function H = getEntropy(x,Bins)
% 
% call:
% 
%       H = getEntropy(x,Bins)
%       
%       
% Computes the entropy of the vector 'x'. Entropy is computed after
% having computed the histogram of the vector, using the binning 
% defined by 'Bins'. 
% if Bins is not specified, the function use the default binning as: 
% Bins = linspace(min(x),max(x),sqrt(numel(x))).
% 
% INPUT
% 
%         x     :   original vector (NOT a distribution !!!)
%         Bins  :   Binning to use to compute the histogram (optional)
%         
%         
% OUTPUT
% 
%         H     :   Entropy of x (a scalar)
%         
%             
% R. G. Bettinardi
if nargin<2
    Bins = linspace(min(x),max(x),sqrt(numel(x)));
end
[counts,binCenters] = hist(x,Bins);
binWidth            = diff(binCenters);
binWidth            = [binWidth(end),binWidth];                        % Replicate last bin width for first, which is indeterminate.
nz                  = counts>0;                                        % Index to non-zero bins
p                   = counts(nz)/sum(counts(nz));                      % probability of each bin
H                   = -sum(p.*log(p./binWidth(nz)));                   % Entropy
end

function len= get_length_seq(v)
% This method calculates the length of all sub-sequences where the array x is either True or 1.
% 
%     Examples
%     --------
%     >>> x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
%     >>> _get_length_sequences_where(x)
%     >>> [1, 3, 1, 2]
% 
%     >>> x = [0,True,0,0,True,True,True,0,0,True,0,True,True]
%     >>> _get_length_sequences_where(x)
%     >>> [1, 3, 1, 2]
% 
%     >>> x = [0,True,0,0,1,True,1,0,0,True,0,1,True]
%     >>> _get_length_sequences_where(x)
%     >>> [1, 3, 1, 2]

if isempty(v)
    len=0;
else
     n   = length(v);
    len = zeros(1, ceil(n/2));
    j   = 0;
    k = 1;
    while k <= n
      if v(k)
        a = k;
        k = k + 1;
        while k <= n && v(k)
           k = k + 1;
        end
        j      = j + 1;
        len(j) = k - a;
      end
      k = k + 1;
    end
    len = len(1:j); 
end

end