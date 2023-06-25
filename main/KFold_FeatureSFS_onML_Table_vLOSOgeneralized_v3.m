%% Takes a Kfold input file, and creates epochs for ML and creates features
%Based on CREATE_EPOCHS_2SPSS
%This attempt is to include TSFresh and other features that we can use to improve
%classification accuracy. Uses SFS to choose the features for kfold.

% created 2/18/20
% updated 2/25/20 - added .txt import
% updated 3/11/20 - added Train-Test ability
% updated 6/1/20 - participants times/files are loaded from master CSV  
%VERSION 8/3/20 - This version is for testing Leave-one-subject out CV. To try
%to train a generalized classifier on all but one subject, pull PREDEFINED
%features with SFS and test the model on a unseen subject. This will
%repeat for each participant and average the accuracy of all CV.
% updated 12/23/21 - changed the ML algorhtims so we dont get wrapper bias
% for JBI manuscript (NOTE: added predefined features for the IEEE Access
% paper to compare LOSO against the SFS 10-fold)
%updated 12/28/22 - Added a Feat_Method variable so you can run predetermeined
%features instead of using SFS in the LOSO.

  
%NOTE: for LOSO, must delete all that dont have  respiration (otherwise
%tables will nto concate)
    
% Tor Finseth

clc 
clearvars
close all

%add effect size function, "mes1way", path
addpath('...\lib\hhentschke-measures-of-effect-size-toolbox-3d90ae5')
%Where the Pans-Tompkins files are located
addpath('...\lib')
%Weka_statistics
addpath('...\lib')
%features
addpath('...\lib\catch22-master\wrap_Matlab')
addpath('...\lib\MarcusVollmer_HRV')
addpath('...\lib\MarcusVollmer_HRV\MarcusVollmer-HRV-564b51f')

% Participant_arr= [66,64,63,62,61,57,56,55,54,50,49,47,45,42,41,31,28,27,26,24,23,22,19,18,17,16,15,14,13,11,9,8,5];     % ISS original, removed participants to enhance accuracy
Participant_arr= [56,24,15,14,11,18,16]; %this participant combo was used for the LOSO test in the publication

% USE SFS? Or use 8 prederemined features for the LOSO?
Feat_Method=1; % SFS=0, predetermined=1 

for participant=1:length(Participant_arr) %loop through the participants and extract features, but no SFS yet
    ParID=Participant_arr(participant);
    
%% INPUTS
Epochsize=10; %In seconds, what the length of epochs should be.
sample_fre=125; %Hz
Trim_StartLen=0 ; %sec, trim from beginning of stress trial
Trim_EndLen=0; %sec, trim from beginning of stress trial
Trimtest=0; %Also trim the testing data for validation, 1=True, 0 =False
PlotFreAnalaysis=0; %Plot FFT,Mel-Spectrogram, Scalogram for ECG and NIBP 1=true, 0=false
hout= 0.2; %ratio of how much data to holdout

%Read participant input file
masterfile= ('...\Data\Participant_input_file_ISS.xlsx'); % or "_ISS" or  "_Nback"
opts = detectImportOptions(masterfile);
opts.SelectedVariableNames = [1]; 
ParID_arr = readmatrix(masterfile,opts);
ParIDrow=find(ParID_arr==ParID); %get which row is the ID in
%get filenames for train and test
opts.SelectedVariableNames = [2:3]; 
file_arr = readmatrix(masterfile,opts);
filename_arr =string(file_arr(ParIDrow,:));
%get the class ordering
opts.SelectedVariableNames = [4:5]; 
class_order_arr = readmatrix(masterfile,opts);
class_order_arr =class_order_arr(ParIDrow,:);
class_order_arr=class_order_arr';
try
class_order_arr{1,:} =  strsplit(class_order_arr{1,:},',');
catch
    error('Did you forget to change the masterfile to match your task? see line 67');
end
class_order_arr{2,:} =  strsplit(class_order_arr{2,:},',');
%get the subjctive ratings
opts.SelectedVariableNames = [6:7]; 
sub_arr = readmatrix(masterfile,opts);
subarraytrain =string(sub_arr(ParIDrow,1));
subarray=string(sub_arr(ParIDrow,2));
subarraytrain =  strsplit(subarraytrain ,' ');
subarray =  strsplit(subarray,' ');
subarraytrain= cellfun(@str2num,subarraytrain,'UniformOutput',false);
subarray= cellfun(@str2num,subarray,'UniformOutput',false);
%class times
opts.SelectedVariableNames = [8:9]; 
time_arr = readmatrix(masterfile,opts);
class_times_arr = time_arr(ParIDrow,:);
% class_times_arr2=class_times_arr2';
class_times_arr{1} =  strsplit(class_times_arr{1},' ');
class_times_arr{2} =  strsplit(class_times_arr{2},' ');
class_times_arr=class_times_arr';
class_times_arr{1}= cellfun(@str2num,class_times_arr{1},'UniformOutput',false);
class_times_arr{2}= cellfun(@str2num,class_times_arr{2},'UniformOutput',false);
%The Subjective rateing from the PSTRM
% subarraytrain=[4 4 5]; %PTSRM individual training ratings in LMH order(even if the chronological order is different) %%%%currently not used for anything
% subarray=[3 4 5]; %PTSRM of testing sessions
subjratings=0; %Run the Test with the subjective ratings as ground truth, 1=TRUE, 0=FALSE

%Pick which chronological sessison(s) to test
%you CANNOT select sessions that are not subsequent (i.e. seperated by another session)
%Test_session=[1]; leave empty to do all of the sessions
Test_session=[];

% class_times_arr={
% };

%check for input errors
if length(filename_arr)~=size(class_times_arr,1)
    error('Filename array and ClassTime array are not the same length. Check Inputs.')
end

%pick testing session
%cant use this method because the anova will want discriminate features
% class_order_arr{2,1}={class_order_arr{2,1}{1,Test_session}};
% for i=1:length(Test_session)
%     ctimes{i}={class_times_arr{2,1}{1,Test_session(i)*2-1},class_times_arr{2,1}{1,Test_session(i)*2}};
% end
% class_times_arr{2,1}=[ctimes{:}];

%Trim the training class times 
for i=1:length(class_times_arr{1,:})
    if mod(i,2) == 1
        class_times_arr{1,:}{1,i}=class_times_arr{1,:}{1,i}+Trim_StartLen;
    else
       class_times_arr{1,:}{1,i}=class_times_arr{1,:}{1,i}-Trim_EndLen;  
    end
end
if Trimtest==1
    %Trim the testing class times 
    for i=1:length(class_times_arr{2,:})
        if mod(i,2) == 1
            class_times_arr{2,:}{1,i}=class_times_arr{2,:}{1,i}+Trim_StartLen;
        else
           class_times_arr{2,:}{1,i}=class_times_arr{2,:}{1,i}-Trim_EndLen;  
        end
    end
end


    close all
    filename=filename_arr(1)
    class_times=class_times_arr(1,:);
    class_times=[class_times{:}];
    class_order=class_order_arr(1,:)'; 
    class_order=[class_order{:}]

%% Make sure file is right type.
[T,classes,ECG_col,NIBP_col,class_dpts] = StressTrialFileImport(filename,class_order,class_times);
if any(class_dpts<=0)
    error('you trimmed to much! Reduce the beginning or ending trim.')
end
% delete the RESP, becaues not all participant files have it
            try
                T.RESP =[];
            catch
            end
count = 0;
%Standardize before deriving features? Doesnt really help. Accuracy stays about the same.
% for i=3:1:11
% T{:,i}= zscore(T{:,i});
% end

    %Trim the training trial to only have middle section
%     [T] = TrimTraining(T,classes,sample_fre,Trim_StartLen,Trim_EndLen); %this is wrong, trim per session, not per class
    T_train=T;
    classes_train=classes;

     %% Delete Signals we dont want
 if Feat_Method==1 % dont compute features for predeteremined
    %     Final_Table.SBP_mean=[];
    %     Final_Table.SBP_std=[];
    %     Final_Table.DBP_mean=[]; %not contributing much
    %     Final_Table.DBP_std=[]; %not contributing much
    %     Final_Table.HeartRate_mean=[]; %RF does 5% better wihtout
    %     Final_Table.HeartRate_std=[]; %RF does 5% better wihtout
    %     Final_Table.RMSSD_mean=[]; % 4 percent worse
    %     Final_Table.RMSSD_std=[]; % 4 percent worse
        T.SBP_1=[]; % 3 percent worse ABayes, bet nothing much on everything eslse
       T.DBP_1=[]; 
       T.pNN=[];
       T.RespirationRate=[];
       T.EDAphasic=[];
       T.EDA_Ln=[];
       T.EDAtonic_Ln=[];
       T.pNN50_Ln=[];
    %     Final_Table.DBP_1_std=[]; % 
    %     Final_Table.RespirationRate_mean=[];
    %     Final_Table.RespirationRate_std=[];
    %     Final_Table.pNN_mean=[]; %very slight rise
    %     Final_Table.pNN_std=[];
    %     Final_Table.EDAphasic_mean=[]; % 2-3% worse without
    %     Final_Table.EDAphasic_std=[];
    %     Final_Table.EDAtonic_mean=[]; %4% worse without
    %     Final_Table.EDAtonic_std=[]; %4% worse without
    %         Final_Table.EDA_Ln_mean=[]; %ABayes did 3% better
    %         Final_Table.EDA_Ln_std=[]; %ABayes did 3% better
    %         Final_Table.EDAtonic_Ln_mean=[]; %slighlty worse
    %         Final_Table.EDAtonic_Ln_std=[]; %slighlty worse
    %         Final_Table.pNN50_Ln_mean=[]; %slighlty worse
    %         Final_Table.pNN50_Ln_std=[];%slighlty worse
 end
    
    
%% Reduce the data and create new features
for k=1:length(classes) %loop for each class

%opts = detectImportOptions(filename);
%T = readtable(filename,opts);
% determine length of table, 125hz*60seconds = 7500 data poitns per epoch
[M,N]=size(T);
%G= findgroups(T.class);
[~,~,G] = unique(string(T.class),'stable');
numClass = splitapply(@numel,T.class,G);
%[classes] = unique(T.class,'stable');

epochpoints=sample_fre*Epochsize;
numepochs=floor(numClass/epochpoints);              %number of epochs per class
for h=1:length(numepochs)
        if numepochs(h)<=1
            numepochs(h)=1;
        end
end

remainder=rem(numClass,epochpoints);
    if size(numClass,1)==3
        Trainlastdpt=[numClass(1),numClass(1)+numClass(2),sum(numClass)]; 
    else
        Trainlastdpt=[numClass(1),sum(numClass)];
    end

% average the DVs for each epoch and save in new array
for i=1:N-1 %loop for each DV, but not class column
    if k==1
        lastdpt=0;
    end
    if k==2
        lastdpt=numClass(k-1);
    end   
    if  k==3
        lastdpt=numClass(k-1)+numClass(k-2);
    end   
    for j=1:numepochs(k) %loop for each epoch
        if j~=numepochs(k)
            firstdpt=lastdpt+1;
            lastdpt=epochpoints+firstdpt-1;
        else
            firstdpt=lastdpt+1;
            lastdpt=epochpoints+firstdpt-1+remainder(k); %for last epoch, add remainder
        end
        y=T{firstdpt:lastdpt,i};
        
            %Calculate the features for the DV segment
           
        if Feat_Method==0 % dont compute features for predeteremined
            if i==1 %Only calculate once, not for every DV column
                 [tsffeat{j,i},tsffeat_names]=tsfreshMATLAB(y,sample_fre);
                [HRVfeatures{j,i},HRVfeatures_names] = GetHRVfeatures(j,T,sample_fre,firstdpt,lastdpt,ECG_col,NIBP_col,PlotFreAnalaysis);
            end
        else
             [tsffeat{j,i},tsffeat_names]=tsfreshMATLAB2(y,sample_fre);
%             M= mean(y,1);
%             stddev= std(y,1);
%             tsffeat{j,i}={M,stddev};
%             tsffeat_names={'mean','std'};
        end
    end
end

% epochavg=num2cell(epochavg)
classlabel(1:numepochs(k),1)=classes(k);
index=[1:numepochs(k)]';

%% Add labels to the tfresh features

for q=1:N-1
    flag=0; %create a table on first iter, then join for everyiter after
    if (q==1)
%         T2=table(index,vertcat(tsffeat{:,1}),'VariableNames',[cellstr("index") cellfun(@(x) join([T.Properties.VariableNames(1),"_",x],''),tsffeat_names)]);
        T2=cell2table(vertcat(tsffeat{:,1}),'VariableNames',[cellfun(@(x) join([T.Properties.VariableNames(1),"_",x],''),tsffeat_names)]);
        T2 = addvars(T2,index);
        flag=1;
    end
%     temp=table(index,vertcat(tsffeat{:,q}),'VariableNames',[cellstr("index") cellfun(@(x) join([T.Properties.VariableNames(1),"_",x],''),tsffeat_names)]);
    temp=cell2table(vertcat(tsffeat{:,q}),'VariableNames',[cellfun(@(x) join([T.Properties.VariableNames(q),"_",x],''),tsffeat_names)]);
    temp = addvars(temp,index);
    if flag==0
        T2=join(T2,temp,'Keys','index'); 
    end
end
tsffeat=[];
    if Feat_Method==0 %no HRV features were made for predeteremined LOSO
    %% Add labels to the HRV features
    %Join the avgHR and add the feature names
    % for q=1:length(HRVfeatures)
        temp=cell2table(vertcat(HRVfeatures{:,1}),'VariableNames',HRVfeatures_names);
        %temp=table(index,cell2mat(HRVfeatures(q)),'VariableNames',[cellstr("index") HRVfeatures_names(q)]);
        temp = addvars(temp,index);
        T2=join(T2,temp,'Keys','index'); 
    % end
    HRVfeatures=[];
    end
%Add the Class Label
temp=table(index,classlabel,'VariableNames',[cellstr("index") T.Properties.VariableNames(N)]);
T2=join(T2,temp,'Keys','index');

count = count + 1;
eval(['T_C' num2str(count) ' = T2;']);

fprintf('T_C%0i created...\n',count);
clearvars avgHR rmssd pNN50 pLF pHF LFHFratio VLF LF HF SDSD SDNN pNN20 TRI TINN SD1 SD2 SD1SD2ratio classlabel p numClass MASS_class
end

%% Combine the class tables
[filepath,name,ext] = fileparts(filename);
ID = str2num(extractBetween(name,'P','_'));
if length(classes)==1
    Final_Table = [T_C1];
elseif length(classes)==2
    Final_Table = [T_C1;T_C2];
elseif length(classes)==3
    Final_Table = [T_C1;T_C2;T_C3];
end
ID_array = repelem(ID,size(Final_Table,1))'; 
Final_Table.ParID=ID_array;
Final_Table = movevars(Final_Table,'index','Before',1);
Final_Table = movevars(Final_Table,'class','Before','index');
Final_Table = movevars(Final_Table,'ParID','Before','class');
%Add standardized data
% col=size(Final_Table,2);
% for k=4:col
%     mean_col=mean(Final_Table{:,k});
%     std_col=std(Final_Table{:,k});
%     st_data=(Final_Table{:,k}-mean_col)/std_col;
%     Final_Table=addvars(Final_Table,st_data,'NewVariableNames',strcat(Final_Table.Properties.VariableNames(k),'std'));
% end
%delete ECG column
    count=1;
    for N=1:size(Final_Table,2)
        if find(string(Final_Table.Properties.VariableNames{1,N}(1:2))=="EC")==1  
            extra(count) = N;
            count=count+1;
        end
    end
    Final_Table(:,extra)=[];
%delete NIBP column
    count=1;
    for N=1:size(Final_Table,2)
        if find(string(Final_Table.Properties.VariableNames{1,N}(1:2))=="NI")==1  
            extra(count) = N;
            count=count+1;
        end
    end
    Final_Table(:,extra)=[];
%delete RESP column
%     count=1; 
%     extra=[];
%     for N=1:size(Final_Table,2)
%         if find(string(Final_Table.Properties.VariableNames{1,N}(1:2))=="RE")==1  
%             extra(count) = N;
%             count=count+1;
%         end
%     end
%     if isempty(extra)==0
%         Final_Table(:,extra)=[];
%     end
%Delete unneeded variables(columns), but not ECG
%     count=1;
%     for N=1:length(Final_Table.Properties.VariableNames)
%         if find(string(Final_Table.Properties.VariableNames{1,N}(1:2))=="se")==1  
%             extra(count) = N;
%             count=count+1;
%         end
%     end
%     Final_Table(:,extra)=[];
    %Final_Table.secMean = [];
%     Final_Table.ZXNICORMean = [];
%     Final_Table.RSPXRSPECRMean = [];
%     Final_Table.NIBPDNoninvasiveBloodPressureDACMean = [];
%     Final_Table.dZdtXNICORMean = [];
%     Final_Table.EDAYPPGEDRMean =[];
%     Final_Table.NIBPMean=[];
if Feat_Method==1 % dont compute features for predeteremined
%     Final_Table.SBP_mean=[];
%     Final_Table.SBP_std=[];
%     Final_Table.DBP_mean=[]; %not contributing much
%     Final_Table.DBP_std=[]; %not contributing much
%     Final_Table.HeartRate_mean=[]; %RF does 5% better wihtout
%     Final_Table.HeartRate_std=[]; %RF does 5% better wihtout
%     Final_Table.RMSSD_mean=[]; % 4 percent worse
%     Final_Table.RMSSD_std=[]; % 4 percent worse
%     Final_Table.SBP_1_mean=[]; % 3 percent worse ABayes, bet nothing much on everything eslse
%     Final_Table.SBP_1_std=[]; % 3 percent worse ABayes, bet nothing much on everything eslse
%     Final_Table.DBP_1_mean=[]; 
%     Final_Table.DBP_1_std=[]; % 
%     Final_Table.RespirationRate_mean=[];
%     Final_Table.RespirationRate_std=[];
%     Final_Table.pNN_mean=[]; %very slight rise
%     Final_Table.pNN_std=[];
%     Final_Table.EDAphasic_mean=[]; % 2-3% worse without
%     Final_Table.EDAphasic_std=[];
%     Final_Table.EDAtonic_mean=[]; %4% worse without
%     Final_Table.EDAtonic_std=[]; %4% worse without
%         Final_Table.EDA_Ln_mean=[]; %ABayes did 3% better
%         Final_Table.EDA_Ln_std=[]; %ABayes did 3% better
%         Final_Table.EDAtonic_Ln_mean=[]; %slighlty worse
%         Final_Table.EDAtonic_Ln_std=[]; %slighlty worse
%         Final_Table.pNN50_Ln_mean=[]; %slighlty worse
%         Final_Table.pNN50_Ln_std=[];%slighlty worse
end
    %Final_Table.RespirationRateMean=[];

%% CREATE table for the individual
[m,n]=size(Final_Table);
ML_Table=movevars(Final_Table,'class','After',n);
% ML_Table=ML_Table(:,3:n);
Vnames = ML_Table.Properties.VariableNames(3:n);

%% Concate a raw data table for multiple participants
        if exist('ML_rawTable','var')==0 %concate the ML_Table for multiple participants
            ML_rawTable=ML_Table;
        else
             ML_rawTable= [ML_rawTable; ML_Table];
        end
end
ML_TableBefore=ML_rawTable;
%%remove features
% ML_rawTable.EDAtonic_c3=[]; %RF=58.35
% ML_rawTable.EDAtonic_abs_energy=[];
% ML_rawTable.EDAtonic_longest_strike_below=[];
% ML_rawTable.EDAtonic_longest_strike_above=[];
% ML_rawTable.EDAtonic_cid_ce=[];
% ML_rawTable.EDAtonic_abs_sum_changes=[];

ML_rawTable.HeartRate_abs_sum_changes=[];
ML_rawTable.DBP_stddev=[];
ML_rawTable.HeartRate_abs_energy=[];
ML_rawTable.DBP_sum_reoccuring_dpt=[];
ML_rawTable.DBP_cid_ce=[];
ML_rawTable.DBP_longest_strike_below=[];
ML_rawTable.DBP_sum_reoccuring_val=[];
ML_rawTable.HeartRate_longest_strike_above=[];
ML_rawTable.DBP_skew=[];
ML_rawTable.RMSSD_abs_energy=[];
ML_rawTable.DBP_longest_strike_above=[];
ML_rawTable.HeartRate_sum_reoccuring_dpt=[];
ML_rawTable.SBP_sum_reoccuring_dpt=[];
ML_rawTable.EDAtonic_abs_energy=[];
ML_rawTable.SBP_abs_sum_changes=[];
ML_rawTable.EDAtonic_longest_strike_above=[];


count=0;
%% Create Loop to test the accuracy of different features removed
ML_TableOr=ML_rawTable;
% for feat_removed=1:size(ML_rawTable,2)-3
%     %remove each feature iteratively
%     ML_rawTable=ML_TableOr;
%     col_removed=feat_removed+2;
%     ML_rawTable(:,col_removed)=[];

%% LOSO on the table
for loso=1:length(Participant_arr)   
    so=Participant_arr(loso);
    Par_training=setdiff(Participant_arr,so,'stable');
%     clearvars ML_TableTrain
    %cleanup from last participant loop
    mu_train=[]; %cleanup
    sigma_train=[]; %cleanup  
%     try %if it errors because not enought testing data, then just skip  the participant


%% Classify, & Compile Fstat and FeatureID in array
    %only standardize the training dataset, dont standardize the subject out
    %create unstandardized table for the SO testing dataset
     ML_Table_unstd=ML_rawTable(ML_rawTable.ParID==so,:); 
     ML_Table_unstd=ML_Table_unstd(:,3:end);
    %Format the training table
     ML_Table=ML_rawTable(ML_rawTable.ParID~=so,:);
     ML_Table=ML_Table(:,3:end);
    %Standardize the training ML_table wrt. itself
    Std_ML_Table=[]; %cleanup
    for i=1:size(ML_Table,2)-1
        [Std_ML_Table{i},mu_train(i),sigma_train(i)]= zscore(ML_Table{:,i});%Standardize ML_table wrt. itself 
    end
    table_class=ML_Table(:,end);
    Std_ML_Table=table(Std_ML_Table{1:size(ML_Table,2)-1});
    Std_ML_Table.Properties.VariableNames = ML_Table.Properties.VariableNames(1:size(ML_Table,2)-1);
    ML_TableTrain=[Std_ML_Table table_class];
    



%At this point, before SFS, there should only be two tables:
%ML_TableTrain= all the paricipants-1 and their features
%ML_TableTest=one particiant and their features
%% Run Sequential Feature Selection on the training datasets of all the participants (except one)

% addpath('C:\Users\tfinseth\Documents\MATLAB\Add-Ons\Collections\SVM-RFE');

%Prep Traning data for SFS
[~,n]=size(ML_TableTrain);
Ctable = table2cell(ML_TableTrain);
Std_dataTrain = cell2mat(Ctable(:,1:n-1));
feature_VariableNames=ML_TableTrain.Properties.VariableNames(1:n-1);
class_labels=[Ctable{:,n}]';

% if LP==2
    ML_TableTest=ML_Table_unstd;
    %Scale the testing dataset based on the standardized training datatset
    for i=1:size(ML_Table_unstd,2)-1
        ML_TableTest{:,i}=rescale(ML_Table_unstd{:,i},min(ML_TableTrain{:,i}),max(ML_TableTrain{:,i}));
    end
    %standardize TableTest wrt TableTrain
    for i=1:size(ML_Table_unstd,2)-1
        ML_TableTest{:,i}=(ML_Table_unstd{:,i}-mu_train(i))/sigma_train(i);
    end
    %Prep Test data for later
    [~,n]=size(ML_TableTest);
    Ctable2 = table2cell(ML_TableTest);
    Std_dataTest = cell2mat(Ctable2(:,1:n-1));
    class_labels2=[Ctable2{:,n}]';
% else
%     Std_dataTest=Std_dataTrain; % Not used, just so I can get CV results fro trial1
%     class_labels2=class_labels;
% end

% Perform SFS, sequentialfs creates candidate feature subsets by sequentially adding each of the features not yet selected.
hout=.2; %ratio of holdout
SFSfold=10; %number of folds in discriminant analysis
iteration=1;
for algor=1:4 %iterate for each ML function
%         try
            if algor==1
                disp("ABayes")
            elseif algor==2
                disp("SVM")
            elseif algor==3
                disp("DT")
            else
                disp("RF")
            end
            if Feat_Method==0 % dont do SFS for predeteremined LOSO
                [newF] = IterativeSequentialfs_v2(Std_dataTrain,class_labels,hout,SFSfold,iteration,algor)
            else
                [newF] = [1:size(Std_dataTrain,2)];
            end
 
%%%%%%%%%%%%%%% Select features to output to K-fold & Train-Test %%%%%%%%%%%%%%%%%%%
%find how many features
newML_arrTrain=[];
newML_arrTest=[];
for i=1:length(newF)
      newML_arrTrain=[newML_arrTrain Std_dataTrain(:,newF(i))];
      newML_arrTest=[newML_arrTest Std_dataTest(:,newF(i))];
end
temp1=array2table(newML_arrTrain,'VariableNames',feature_VariableNames(newF));
temp2=array2table(newML_arrTest,'VariableNames',feature_VariableNames(newF));

  %create tables with features from SFS
        if algor==1 %ABayes
            ML_TableTrainAB = addvars(temp1,class_labels,'after',size(temp1,2));
            ML_TableTrainAB.Properties.VariableNames{'class_labels'} = 'class';
            ML_TableTestAB = addvars(temp2,class_labels2,'after',size(temp2,2));
            ML_TableTestAB.Properties.VariableNames{'class_labels2'} = 'class';
            FS_namesAB=feature_VariableNames(newF); %names features selected
        end
        if algor==2 %SVM
            ML_TableTrainSVM = addvars(temp1,class_labels,'after',size(temp1,2));
            ML_TableTrainSVM.Properties.VariableNames{'class_labels'} = 'class';
            ML_TableTestSVM = addvars(temp2,class_labels2,'after',size(temp2,2));
            ML_TableTestSVM.Properties.VariableNames{'class_labels2'} = 'class';
            FS_namesSVM=feature_VariableNames(newF); %names features selected
        end
        if algor==3 %DT
            ML_TableTrainDT = addvars(temp1,class_labels,'after',size(temp1,2));
            ML_TableTrainDT.Properties.VariableNames{'class_labels'} = 'class';
            ML_TableTestDT = addvars(temp2,class_labels2,'after',size(temp2,2));
            ML_TableTestDT.Properties.VariableNames{'class_labels2'} = 'class';
            FS_namesDT=feature_VariableNames(newF); %names features selected
        end
        if algor==4 % RF
            ML_TableTrainRF = addvars(temp1,class_labels,'after',size(temp1,2));
            ML_TableTrainRF.Properties.VariableNames{'class_labels'} = 'class';
            ML_TableTestRF = addvars(temp2,class_labels2,'after',size(temp2,2));
            ML_TableTestRF.Properties.VariableNames{'class_labels2'} = 'class';
            FS_namesRF=feature_VariableNames(newF); %names features selected
        end
end
%create tables with features from SFS
% ML_TableTrain = addvars(temp1,class_labels,'after',size(temp1,2));
% ML_TableTrain.Properties.VariableNames{'class_labels'} = 'class';
% ML_TableTest = addvars(temp2,class_labels2,'after',size(temp2,2));
% ML_TableTest.Properties.VariableNames{'class_labels2'} = 'class';

%% Classify using LOSO (was previously designed to testing each fold of the held out participant, but now just testing the entire participant
LP=1;
FS_names(1)={feature_VariableNames(newF)}; %names features selected


set(gcf,'Visible','off');  %supress plots            
set(0,'DefaultFigureVisible','off');

%ground truth
 actual=ML_TableTestSVM{:,end};

  %ABayes
  features=ML_TableTestAB.Properties.VariableNames(1:end-1);
 [classname,~,~] = unique(ML_TableTrainAB.class,'stable');
 [~,~,framepred,~,~,~, c_matrix_AB] = kernal_function_Test_Realtime3(ML_TableTestAB,ML_TableTrainAB,features);
 [acc_ABayes,Abayes_ttmetrics]=Weka_statistics(c_matrix_AB,class_order);
 
 %%NOTE: This code was here previoulsy for the working LOSO, replaced it with new code that prevents data leakage
% [acc2f(LP), acc2fmetrics] = kernal_cross_batch3_function2(ML_TableTrain,2); %ABayes with Random Stratified Partition
% %If there are too few datapoints, the CM doesnt get fully created, causing errors.
% try
%     [acc10f(LP), acc10fmetrics]  = kernal_cross_batch3_function2(ML_TableTrain,10); %ABayes with Random Stratified Partition
% catch
%    warning('Not eough datapoints to complete 10-fold CV for ABayes. One or more folds do not contain points from all the groups. ')
%    acc10f(LP) =NaN;
%    acc10fmetrics=[NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN];
% end


%https://www.mathworks.com/help/stats/fitcecoc.html, 'Learners'
%SVM
y=ML_TableTrainSVM{:,end};
x=ML_TableTrainSVM{:,1:end-1};
SVMModel = fitcecoc(x,y,'Learners', 'svm');
[label, ~] = predict(SVMModel,ML_TableTestSVM{:,1:end-1});%Test
actual=ML_TableTestSVM{:,end};
label=string(label);
[c_matrix_SVM]= confusionmat(actual,label,'Order',{'L' 'M' 'H'});
[acc_SVM,SVM_ttmetrics]=Weka_statistics(c_matrix_SVM,class_order);

%Tree
y=ML_TableTrainDT{:,end};
x=ML_TableTrainDT{:,1:end-1};
SVMModel = fitcecoc(x,y,'Learners', 'tree');
[label, ~] = predict(SVMModel,ML_TableTestDT{:,1:end-1});%Test
actual=ML_TableTestDT{:,end};
label=string(label);
[c_matrix_DT]= confusionmat(actual,label,'Order',{'L' 'M' 'H'});
[acc_DT,DT_ttmetrics]=Weka_statistics(c_matrix_DT,class_order);

%Random Forest
y=ML_TableTrainRF{:,end};
x=ML_TableTrainRF{:,1:end-1};
RFModel = fitcensemble(x,y,'Method','Bag');
[label, ~] = predict(RFModel,ML_TableTestRF{:,1:end-1});%Test
actual=ML_TableTestRF{:,end};
label=string(label);
[c_matrix_RF]= confusionmat(actual,label,'Order',{'L' 'M' 'H'});
[acc_RF,RF_ttmetrics]=Weka_statistics(c_matrix_RF,class_order);


% disp(' ')
%  disp('%-------- LOSO Accuracy for single participant -----------%')
%  fprintf('ABayes                                %0.2f %%\n',acc_ABayes(1)*100);
%  fprintf('SVM                                    %0.2f %%\n',acc_SVM(1)*100);
%   fprintf('DT                                      %0.2f %%\n',acc_DT(1)*100);
%   fprintf('RF                                      %0.2f %%\n',acc_RF(1)*100);

%% Compile participant data into one array
PR_completed(loso,1)=so;

%Train-test (testing on the entire participant that was held out)
Par_Abayes_tt(loso,1)=acc_ABayes(LP);
Par_SVM_tt(loso,1)=acc_SVM(LP);
Par_DT_tt(loso,1)=acc_DT(LP);
Par_RF_tt(loso,1)=acc_RF(LP);

Par_Abayes_ttmetrics(loso,:)=Abayes_ttmetrics;
Par_SVM_ttmetrics(loso,:)=SVM_ttmetrics;
Par_DT_ttmetrics(loso,:)=DT_ttmetrics;
Par_RF_ttmetrics(loso,:)=RF_ttmetrics;

%Get SFS features for each participant
Par_SFSfeatures(loso,1)={newF'};
Par_SFSnames(loso,1)=FS_names;

%for parrallel plot
if exist('agg_actual','var')==0 %concate the ML_Table for multiple participants
            agg_actual=actual;
        else
             agg_actual=[agg_actual; actual];
end
tab=table2array(ML_TableTestRF);
tab2=str2double(tab(:,:));
if exist('agg_stdtab','var')==0 %concate the ML_Table for multiple participants
            agg_stdtab=tab2;
        else
             agg_stdtab=[agg_stdtab; tab2];
end
count=count+1
        
end %end batch loop
%This avaerages each of the loso CVs
%combine arrays for excel
% Par_exceltt{feat_removed}=[PR_completed,Par_Abayes_tt,Par_SVM_tt,Par_DT_tt,Par_RF_tt];
% Par_excelttm{feat_removed}=[PR_completed,Par_Abayes_ttmetrics,Par_SVM_ttmetrics,Par_DT_ttmetrics,Par_RF_ttmetrics];
Par_exceltt=[PR_completed,Par_Abayes_tt,Par_SVM_tt,Par_DT_tt,Par_RF_tt];
Par_excelttm=[PR_completed,Par_Abayes_ttmetrics,Par_SVM_ttmetrics,Par_DT_ttmetrics,Par_RF_ttmetrics];

%enable plots again
set(gcf,'Visible','on');              
set(0,'DefaultFigureVisible','on');

%get the list of choosen FS from SFS
% ListnamesFS=[Par_SFSnames{:}];
% [ListnamesFSUn,~,idx] = unique(ListnamesFS,'stable');
% [ListnamesFSUnSort,ListIdxSortbycount]=sort(groupcounts(idx),'descend');
% 
% figure;
% bar(1:length(ListIdxSortbycount),ListnamesFSUnSort);
% grid on
% title('SFS - most selected features for participants')
% xlabel('Features selected by SFS')
% ylabel('Number of Paricipants using feature')
% xticks(1:length(ListnamesFSUn))
% xticklabels(strrep(ListnamesFSUn(ListIdxSortbycount),'_','\_'))
% xtickangle(45)

disp(' ')
 disp('%-------- LOSO Accuracy Averaged -----------%')
 fprintf('ABayes                                %0.2f %%\n',mean(Par_Abayes_tt)*100);
 fprintf('SVM                                    %0.2f %%\n',mean(Par_SVM_tt)*100);
  fprintf('DT                                      %0.2f %%\n',mean(Par_DT_tt)*100);
  fprintf('RF                                      %0.2f %%\n',mean(Par_RF_tt)*100);
% end %%%%%%%%%%%%%%%%%%%  feature iteration loop
 agg_stdtab2=agg_stdtab(:,1:end-1);
ListnamesFSUn2= convertCharsToStrings(ListnamesFSUn);
figure;
parallelcoords(agg_stdtab2,'Group',agg_actual,'Labels',ListnamesFSUn2,'quantile',.25,'LineWidth',2)

%Normalize to plot the parrallel coordinates
for i=1:size(agg_stdtab2,2)
agg_stdtab2norm(:,i) = (agg_stdtab2(:,i) - min(agg_stdtab2(:,i))) / ( max(agg_stdtab2(:,i)) - min(agg_stdtab2(:,i)) );
end

figure;
parallelcoords(agg_stdtab2norm,'Group',agg_actual,'Labels',ListnamesFSUn2)

for i=1:size(Par_exceltt,2)
    mat=Par_exceltt{1,i};
    M=mean(mat);
    Agg_Acc(i,:)=M(1,2:5)
end         
