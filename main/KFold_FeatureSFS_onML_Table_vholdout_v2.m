%% Takes a Kfold input file, and creates epochs for ML and creates features
%Based on CREATE_EPOCHS_2SPSS
%This attempt is to include TSFresh and other features that we can use to improve
%classification accuracy. Uses SFS to choose the features for kfold.

% See ITERATIVESEQUENTIALFS.m if you want to include or remove correlate features (uncorr_feat).
% See kernal_function_Test_Realtime.m to change if weight should be 33% or by class size (margin).

%MATLAB 2019b

% created 2/18/20
% updated 2/25/20 - added .txt import
% updated 3/11/20 - added Train-Test ability
% updated 6/1/20 - participants times/files are loaded from master CSV  
% updated 6/9/20 - THIS VERSION IS FOR HOLDOUT ONLY (the kfold and test-train have been gutted)
% updated 6/16/21 - changed the ML algorhtims so we dont get wrapper bias for JBI manuscript
    
% Tor Finseth

clc 
clearvars
close all

%add effect size function, "mes1way", path
addpath('...\hhentschke-measures-of-effect-size-toolbox-master')
%Where the Pans-Tompkins files are located
addpath('...\lib')
%Where HRV alorthigms are located
 addpath('...\lib\MarcusVollmer_HRV\MarcusVollmer-HRV-564b51f')
%Weka_statistics
addpath('...\lib')
%features
addpath('...\lib\catch22-master\wrap_Matlab')

%Select the participants you want to run (note: larger window sizes cause some participants to error. Below are suggested participant combinations. Change 'Epochsize' to match selection)
%Note: Id=12 removed, becuse not all data recorded for ISS-High
% VR-ISS----------
% Participant_arr= [66,64,63,62,61,57,56,55,54,52,51,50,49,47,45,43,42,41,31,29,28,27,26,24,23,22,19,18,17,16,15,14,13,11,9,8,5];     % ISS original, removed 42,12,29 for 20sec, 49,15 for 30sec, 21 for 40 sec, 47 for 30-60sec, Holdout 56,54,43,34 for 20,30sec (NF), 27,19 for 30sec, 66,57,54,52,45,41,18 for 40secc ---
% Participant_arr= [66,64,63,62,61,57,56,55,54,52,51,50,49,47,45,43, 41,28,24,23,22,19,18,16,15,13,9];     % ISS 10-sec,
% Participant_arr= [66,64,63,62,61,57,55,54,52,51,50,49,45,41,28,24,23,22,16,13,9];     % ISS 20-sec,  ---19,18 deleted
% Participant_arr= [64,63,62,61,55, 50,49,24,23,22,16,13,9];     % ISS 30-sec,  --66,57,54,52,51, 45,41,28, deleted 
% Participant_arr= [64,63,61,23,16,13,9];     % ISS 40-sec, 62, 51,50, 28,24,22, deleted 
%Nback------------
% Participant_arr= [68,67,39,38,37,36,29,27,26,17,14,11,8,5]; % Nback 10 seconds
% Participant_arr= [68,67,39,38,37,36,29,27,26,17,14,11,8,5]; % Nback 20 seconds
Participant_arr= [39,38,37,36,29,27,26,17,14,11,8,5]; % Nback 30 seconds
% Participant_arr= [39,38,37,36,27,26,17,14,11,5]; % Nback 40 seconds

TT=2;% Holdout=2, Train-test=1, k-fold=0 (dont change leave as holdout, otherwise code might break)


for participant=1:length(Participant_arr)
    ParID=Participant_arr(participant);
    try
%% INPUTS
Epochsize=30; %In seconds, what the length of epochs should be.
sample_fre=125; %Hz
Trim_StartLen=0; %sec, trim from beginning of stress trial
Trim_EndLen=0; %sec, trim from beginning of stress trial
Trimtest=0; %Also trim the testing data for validation, 1=True, 0 =False
PlotFreAnalaysis=0; %Plot FFT,Mel-Spectrogram, Scalogram for ECG and NIBP 1=true, 0=false
hout= 0.2; %ratio of how much data to holdout

%Read participant input file
masterfile= ('...\Data\Participant_input_file_Nback.xlsx'); % or "_ISS" or  "_Nback" 
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
class_order_arr{1,:} =  strsplit(class_order_arr{1,:},',');
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

%check for input errors
if length(filename_arr)~=size(class_times_arr,1)
    error('Filename array and ClassTime array are not the same length. Check Inputs.')
end

%cleanup from last participant loop
    mu_train=[]; %cleanup
    sigma_train=[]; %cleanup
    Std_ML_Table=[]; %cleanup

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
%loop for train-test or k-fold
if TT==1                       
    LParr=1:length(filename_arr);
else                    %only run once for kfold and holdout
    LParr=1; 
end
for LP=LParr
    close all
    filename=filename_arr(LP)
    class_times=class_times_arr(LP,:);
    class_times=[class_times{:}];
    class_order=class_order_arr(LP,:)'; 
    class_order=[class_order{:}];

%% Make sure file is right type.
[T,classes,ECG_col,NIBP_col,class_dpts] = StressTrialFileImport(filename,class_order,class_times);
if any(class_dpts<=0)
    error('you trimmed to much! Reduce the beginning or ending trim.')
end
count = 0;
%Standardize before deriving features? Doesnt really help. Accuracy stays about the same.
% for i=3:1:11
% T{:,i}= zscore(T{:,i});
% end
if LP==1
    %Trim the training trial to only have middle section
%     [T] = TrimTraining(T,classes,sample_fre,Trim_StartLen,Trim_EndLen); %this is wrong, trim per session, not per class
    T_train=T;
    classes_train=classes;
else
    T_test=T;
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
if LP==1
    if size(numClass,1)==3
        Trainlastdpt=[numClass(1),numClass(1)+numClass(2),sum(numClass)]; 
    else
        Trainlastdpt=[numClass(1),sum(numClass)];
    end
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
    [tsffeat{j,i},tsffeat_names]=tsfreshMATLAB_welch(y,sample_fre);

    if i==1 %Only calculate once, not for every DV column
        [HRVfeatures{j,i},HRVfeatures_names] = GetHRVfeatures(j,T,sample_fre,firstdpt,lastdpt,ECG_col,NIBP_col,PlotFreAnalaysis);
    end
   
end
    %SDST for training dataset
    if LP==1
        %get matrix profile for each DV in each class
        %[MP] = interactiveMatrixProfileVer3_website(T{:,2},1250); %TOOO SLOWWWW, 12 second for one DV
        %find motifs, take time segments assocaited with motifs
        %[MPSort,MPIdxSort] = sort(MP,'ascend');
        %[val2 loc2]=min(MP);
        %calculate threshold from time segments
        %H=val % or -10% of val
        %use this hreshold to help classify instead of location
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
%% Add labels to the HRV features
%Join the avgHR and add the feature names
% for q=1:length(HRVfeatures)
    temp=cell2table(vertcat(HRVfeatures{:,1}),'VariableNames',HRVfeatures_names);
    %temp=table(index,cell2mat(HRVfeatures(q)),'VariableNames',[cellstr("index") HRVfeatures_names(q)]);
    temp = addvars(temp,index);
    T2=join(T2,temp,'Keys','index'); 
% end
HRVfeatures=[];
%Add the Class Label
temp=table(index,classlabel,'VariableNames',[cellstr("index") T.Properties.VariableNames(N)]);
T2=join(T2,temp,'Keys','index');
%end
count = count + 1;
eval(['T_C' num2str(count) ' = T2;']);

fprintf('T_C%0i created...\n',count);
clearvars avgHR rmssd pNN50 pLF pHF LFHFratio VLF LF HF SDSD SDNN pNN20 TRI TINN SD1 SD2 SD1SD2ratio classlabel p numClass 
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
Final_Table = movevars(Final_Table,'class','Before','index');
Final_Table = movevars(Final_Table,'ParID','Before','class');

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

%% CREATE table to classify
[m,n]=size(Final_Table);
ML_Table=movevars(Final_Table,'class','After',n);
ML_Table=ML_Table(:,3:n);
Vnames = ML_Table.Properties.VariableNames;

%% Holdout portion of data to test on
if TT==2
    %Prep data for holdout
    [~,n]=size(ML_Table);
    Ctable = table2cell(ML_Table);
    class_label=[Ctable{:,n}]';
   
    %holdout - nonrandomized split of data, last 20% used for holdout
%     holdoutCVP = cvpartition(class_label,'holdout',round(hout*length(class_label))); %randomized 
%     dataTrain = Ctable(training(holdoutCVP,1),1:end-1); %randomized 
%     grpTrain = class_label(training(holdoutCVP,1)); %randomized 
    G=groupcounts(ML_Table,'class');
    Glen=table2array(G(:,2));
    Grow=[Glen(1);Glen(1)+Glen(2)-1;sum(Glen)];     %last datapoint of class
    G2=round(Glen*hout);
    for gi=1:3                                                         %make sure there is at least one datapoint from each class in holdout
        if G2(gi)<1
            G2(gi)=1;
        end
    end
     HoldoutSplit=zeros(size(ML_Table,1),1);
     for gi=1:3
        HoldoutSplit(Grow(gi)-G2(gi)+1:Grow(gi))=1; %set the holdout split to ones
     end
     HoldoutSplit=logical(HoldoutSplit);
     TrainingSplit=~HoldoutSplit;
    dataTrain = Ctable(TrainingSplit,1:end-1);  
    grpTrain = class_label(TrainingSplit); 
    ML_Table=cell2table(dataTrain(:,1:size(dataTrain,2)));
    table_class=table(grpTrain);
    ML_Table= [ML_Table, table_class];
    ML_Table.Properties.VariableNames = Vnames;
end

%% Classify, & Compile Fstat and FeatureID in array
    %Standardize the ML_table wrt. itself

        for i=1:size(ML_Table,2)-1
            [Std_ML_Table{i},mu_train(i),sigma_train(i)]= zscore(ML_Table{:,i}); %Standardize ML_table wrt. itself 
        end
        table_class=ML_Table(:,end);
        Std_ML_Table=table(Std_ML_Table{1:size(ML_Table,2)-1});
        Std_ML_Table.Properties.VariableNames = ML_Table.Properties.VariableNames(1:size(ML_Table,2)-1);
        ML_TableTrain= [Std_ML_Table table_class];
        clearvars ML_Table

end %end batch loop
%% Run Sequential Feature Selection on the training datasets

addpath('C:\Users\tfinseth\Documents\MATLAB\Add-Ons\Collections\SVM-RFE');

%Prep Traning data for SFS
[~,n]=size(ML_TableTrain);
Ctable2 = table2cell(ML_TableTrain);
Std_dataTrain = cell2mat(Ctable2(:,1:n-1));
feature_VariableNames=ML_TableTrain.Properties.VariableNames(1:n-1);
class_labels=[Ctable2{:,n}]';

%Prep unstandardized Holdout partition for testing
if TT==2  
    dataTest= Ctable(HoldoutSplit,1:end-1);
    grpTest= class_label(HoldoutSplit);
    ML_TableTest=cell2table(dataTest(:,1:size(dataTest,2)));
    table_class=table(grpTest);
    ML_TableTest= [ML_TableTest, table_class];
    ML_TableTest.Properties.VariableNames = Vnames;
    ML_Table_unstd=ML_TableTest;
    %Scale the testing dataset based on the standardized training datatset
%     for i=1:size(ML_Table_unstd,2)-1
%         ML_TableTest{:,i}=rescale(ML_Table_unstd{:,i},min(ML_TableTrain{:,i}),max(ML_TableTrain{:,i}));
%     end
    %standardize TableTest wrt TableTrain
    for i=1:size(ML_Table_unstd,2)-1
        ML_TableTest{:,i}=(ML_Table_unstd{:,i}-mu_train(i))/sigma_train(i);
    end
    %Prep Test data for later
    [~,n]=size(ML_TableTest);
    Ctable2 = table2cell(ML_TableTest);
    Std_dataTest = cell2mat(Ctable2(:,1:n-1));
    class_labels2=[Ctable2{:,n}]';
else
    Std_dataTest=Std_dataTrain; % Not used, just so I can get CV results fro trial1
    class_labels2=class_labels;
end

% Perform SFS, sequentialfs creates candidate feature subsets by sequentially adding each of the features not yet selected.
hout=.2; %ratio of holdout
fold=10; %number of folds in discriminant analysis
iteration=1;
for algor=1:4 %iterate for each ML function
%         try
            [newF] = IterativeSequentialfs_v2(Std_dataTrain,class_labels,hout,fold,iteration,algor)
%         catch
            %if the data size is too small, dont select any features.
%             newF=[];
%             error('no features were selected, because the SFS errored')
%         end
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
            FS_namesAB={feature_VariableNames(newF)}; %names features selected
        end
        if algor==2 %SVM
            ML_TableTrainSVM = addvars(temp1,class_labels,'after',size(temp1,2));
            ML_TableTrainSVM.Properties.VariableNames{'class_labels'} = 'class';
            ML_TableTestSVM = addvars(temp2,class_labels2,'after',size(temp2,2));
            ML_TableTestSVM.Properties.VariableNames{'class_labels2'} = 'class';
            FS_namesSVM={feature_VariableNames(newF)}; %names features selected
        end
        if algor==3 %DT
            ML_TableTrainDT = addvars(temp1,class_labels,'after',size(temp1,2));
            ML_TableTrainDT.Properties.VariableNames{'class_labels'} = 'class';
            ML_TableTestDT = addvars(temp2,class_labels2,'after',size(temp2,2));
            ML_TableTestDT.Properties.VariableNames{'class_labels2'} = 'class';
            FS_namesDT={feature_VariableNames(newF)}; %names features selected
        end
        if algor==4 % RF
            ML_TableTrainRF = addvars(temp1,class_labels,'after',size(temp1,2));
            ML_TableTrainRF.Properties.VariableNames{'class_labels'} = 'class';
            ML_TableTestRF = addvars(temp2,class_labels2,'after',size(temp2,2));
            ML_TableTestRF.Properties.VariableNames{'class_labels2'} = 'class';
            FS_namesRF={feature_VariableNames(newF)}; %names features selected
        end
end
%% Classify
LP=1;
FS_names(LP)={feature_VariableNames(newF)}; %names features selected

%%%%%%%%%%%KFOLD%%%%%%%%%%%%%%%%%%
%ABayes

set(gcf,'Visible','off');  %supress plots            
set(0,'DefaultFigureVisible','off');

% [acc2f(LP), acc2fmetrics] = kernal_cross_batch3_function2(ML_TableTrain,2); %ABayes with Random Stratified Partition
% %If there are too few datapoints, the CM doesnt get fully created, causing errors.
% % if size(ML_TableTrain,1)>=30
% try
%     [acc10f(LP), acc10fmetrics]  = kernal_cross_batch3_function2(ML_TableTrain,10); %ABayes with Random Stratified Partition
% catch
%     warning('Not eough datapoints to complete 10-fold CV for ABayes. One or more folds do not contain points from all the groups. ')
%     acc10f(LP) =NaN;
%     acc10fmetrics=[NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN];
% end

% %Linear Discriminate Analysis
% y=ML_TableTrain{:,end};
% x=ML_TableTrain{:,1:end-1};
% c = cvpartition(y,'KFold',10); %random partition
% fun = @(xTrain,yTrain,xTest,yTest)(sum(~strcmp(yTest,...
%     classify(xTest,xTrain,yTrain)))); %linear discriminant analysis
% LDArate10(LP) = sum(crossval(fun,x,y,'partition',c))...
%            /sum(c.TestSize);
%        
% %Linear Discriminate Analysis
% %rng(1); %to set seed of the random number generation. Will produce consistent results.
%  y=ML_TableTrain{:,end};
%  x=ML_TableTrain{:,1:end-1};
% % c = cvpartition(y,'KFold',2); %random partition
% % fun = @(xTrain,yTrain,xTest,yTest)(sum(~strcmp(yTest,...
% %     classify(xTest,xTrain,yTrain)))); %linear discriminant analysis
% % LDArate2(LP) = sum(crossval(fun,x,y,'partition',c))...
% %            /sum(c.TestSize);
% 
%  %SVM CV
% SVMModel = fitcecoc(x,y);
% CVSVMModel = crossval(SVMModel,'KFold',10);
% SVMrate10(LP) = kfoldLoss(CVSVMModel);
%  [validationPredictions, validationScores] = kfoldPredict(CVSVMModel);
% confmat=confusionmat(y,validationPredictions);
% [acc,SVMrate10metrics]=Weka_statistics(confmat,class_order);
% 
% SVMModel = fitcecoc(x,y);
% CVSVMModel = crossval(SVMModel,'KFold',2);
% SVMrate2(LP) = kfoldLoss(CVSVMModel);
%  [validationPredictions, validationScores] = kfoldPredict(CVSVMModel);
% confmat=confusionmat(y,validationPredictions);
% [acc,SVMrate2metrics]=Weka_statistics(confmat,class_order);
% 
%  %tree CV       
% treeModel = fitctree(x,y);
% CVSVMModel = crossval(treeModel,'KFold',10);
% DTrate10(LP) = kfoldLoss(CVSVMModel);
%  [validationPredictions, validationScores] = kfoldPredict(CVSVMModel);
% confmat=confusionmat(y,validationPredictions);
% [acc,DTrate10metrics]=Weka_statistics(confmat,class_order);
% 
% treeModel = fitctree(x,y);
% CVSVMModel = crossval(treeModel,'KFold',2);
% DTrate2(LP) = kfoldLoss(CVSVMModel);
% [validationPredictions, validationScores] = kfoldPredict(CVSVMModel);
% confmat=confusionmat(y,validationPredictions);
% [acc,DTrate2metrics]=Weka_statistics(confmat,class_order);
% 
% %Random Forest
% RFModel = fitcensemble(x,y,'Method','Bag');
% CVSVMModel = crossval(RFModel,'KFold',10);
% RFrate10(LP) = kfoldLoss(CVSVMModel);
% [validationPredictions, validationScores] = kfoldPredict(CVSVMModel);
% confmat=confusionmat(y,validationPredictions);
% [acc,RFrate10metrics]=Weka_statistics(confmat,class_order);
% 
% RFModel = fitcensemble(x,y,'Method','Bag');
% CVSVMModel = crossval(RFModel,'KFold',2);
% RFrate2(LP) = kfoldLoss(CVSVMModel);
% [validationPredictions, validationScores] = kfoldPredict(CVSVMModel);
% confmat=confusionmat(y,validationPredictions);
% [acc,RFrate2metrics]=Weka_statistics(confmat,class_order);
% 
% 
%  disp('%-------- CV Training Dataset Accuracy -----------%')
% fprintf('ABayes 2fold                        %0.2f %%\n',acc2f(LP)*100);
% % fprintf('LDA 2fold                         %0.2f %%\n',(1-LDArate2(LP))*100);
% fprintf('SVM 2fold                           %0.2f %%\n',(1-SVMrate2(LP))*100);
% fprintf('DT 2fold                              %0.2f %%\n',(1-DTrate2(LP))*100);
% fprintf('RF 2fold                              %0.2f %%\n',(1-RFrate2(LP))*100);
% disp(' ')
% fprintf('ABayes 10fold                     %0.2f %%\n',acc10f(LP)*100);
% % fprintf('LDA 10fold                       %0.2f %%\n',(1-LDArate10(LP))*100);
% fprintf('SVM 10fold                         %0.2f %%\n',(1-SVMrate10(LP))*100);
% fprintf('DT 10fold                             %0.2f %%\n',(1-DTrate10(LP))*100);
% fprintf('RF 10fold                              %0.2f %%\n',(1-RFrate10(LP))*100);
% 
% %Adjust the test dataset labels to be subjective stress ratings
% if subjratings==1
%     LP=2;
% else
%     LP=1; 
% end

%Run the normal ground truth, and then the subjective rating ground truth
for loop=1:LP

 actual=ML_TableTestSVM{:,end};
 actualstressor=actual;
        
 if TT==2  %holdout

 %ABayes
 features=ML_TableTestAB.Properties.VariableNames(1:end-1);
 [classname,~,~] = unique(ML_TableTrainAB.class,'stable');
 [acc_ABayes,nDVsout,framepred,metrics_ABayes] = kernal_function_Test_Realtime(ML_TableTestAB,ML_TableTrainAB,features);

% Plot where the errors are occuring
figure;
x= 1:size(ML_TableTestAB{:,end},1);%time
[C,~,y] = unique(ML_TableTestAB{:,end},'stable'); %classes in test dataset
framepred=[framepred{:}]';
%new code 4/12/20
zeroy=zeros([length(y),1]);
zeroy2=zeros([length(framepred),1]);
zeroy2(find(strcmp(framepred,'L')))=1;
zeroy2(find(strcmp(framepred,'M')))=2;
zeroy2(find(strcmp(framepred,'H')))=3;
zeroy(find(strcmp(ML_TableTestAB{:,end},'L')))=1;
zeroy(find(strcmp(ML_TableTestAB{:,end},'M')))=2;
zeroy(find(strcmp(ML_TableTestAB{:,end},'H')))=3;

    
%plot the actual/prediction graph
smcoeff=6/size(x,2);
f=smooth(x',zeroy2,smcoeff,'lowess'); %local weighted regression smoothing, first degree polynomial  
figure;
plot(x,zeroy,'b',x,zeroy2,'r',x,f,'k')
ylim([0 4])
set(gca, 'YTick',1:3, 'YTickLabel',{'L' 'M' 'H'})%{C(1) C(2) C(3)})
ylabel('Class')
xlabel('datapoint')
legend('stressor','predicted stress','lowess')
cfilepath=char(filepath);
ptit=', ABayes testing dataset';
labe=strcat(cfilepath(end-2:end),ptit);
title(labe)

sess_ep=class_dpts'/sample_fre/Epochsize;
addep=zeros(1,size(class_order,2));
for i=1:size(class_order,2)-1                                           %fix the fact the epochs are based on combined similar trials (e.g., trial 1=5.7epochsL, trial2=4.4epochsL, to total Lepcohs=10)
    sim=strcmp(class_order(i),class_order(i+1));
    if sim==1 && sess_ep(i)-floor(sess_ep(i))+sess_ep(i+1)-floor(sess_ep(i+1))>1
        addep(i)=1;
    end
end
sess_epf=floor(sess_ep)+addep';                                 %epochs in each trial
if size(sess_epf,1)==size(numepochs,1)
    sess_epf=numepochs;
end
if subjratings==1
    zerosub=[];
    for i=1:length(subarray)
        zerosub=[zerosub;repmat(subarray(i),sess_epf(i),1)];
    end
    zerosub=cell2mat(zerosub);
end
%plot the subjective stress
figure;
yyaxis left
plot(x,zeroy,'b',x,zeroy2,'-r',x,f,'-k')
ylim([0 4])
set(gca, 'YTick',1:3, 'YTickLabel',{'L' 'M' 'H'})%{C(1) C(2) C(3)})
ylabel('Class')
xlabel('datapoint')
cfilepath=char(filepath);
ptit=', ABayes testing dataset';
labe=strcat(cfilepath(end-2:end),ptit);
title(labe)
if subjratings==1
%     yyaxis right
%     plot(x,zerosub/3,'color',[0 0.6 0.7410]);
%     ylim([0 4])
%     set(gca, 'YTick',1:3, 'YTickLabel',{3 6 9})
    legend('stressor','predicted stress','lowess','subjective stress')
else
    legend('stressor','predicted stress','lowess')
end

%https://www.mathworks.com/help/stats/fitcecoc.html, 'Learners'
%SVM
y=ML_TableTrainSVM{:,end};
x=ML_TableTrainSVM{:,1:end-1};
SVMModel = fitcecoc(x,y,'Learners', 'svm');
[label, score] = predict(SVMModel,ML_TableTestSVM{:,1:end-1});%Test
actual=ML_TableTestSVM{:,end};
label=string(label);
[c_matrix]= confusionmat(actual,label);
[acc_SVM,metrics_SVM]=Weka_statistics(c_matrix,[classname])

%LDA -diagquadratic
% t = templateDiscriminant('DiscrimType','diagquadratic');
% SVMModel = fitcecoc(x,y,'Learners', t);
% [label, score] = predict(SVMModel,ML_TableTest{:,1:end-1});%Test
% actual=ML_TableTest{:,end};
% label=string(label);
% [c_matrix]= confusionmat(actual,label,'ORDER',{'L','M','H'});
% acc_LDA(loop)=Weka_statistics(c_matrix,[classname])

%Tree
y=ML_TableTrainDT{:,end};
x=ML_TableTrainDT{:,1:end-1};
SVMModel = fitcecoc(x,y,'Learners', 'tree');
[label, score] = predict(SVMModel,ML_TableTestDT{:,1:end-1});%Test
actual=ML_TableTestDT{:,end};
label=string(label);
[c_matrix]= confusionmat(actual,label);
[acc_DT,metrics_DT]=Weka_statistics(c_matrix,[classname])

%Random Forest
y=ML_TableTrainRF{:,end};
x=ML_TableTrainRF{:,1:end-1};
RFModel = fitcensemble(x,y,'Method','Bag');
[label, score] = predict(RFModel,ML_TableTestRF{:,1:end-1});%Test
actual=ML_TableTestRF{:,end};
label=string(label);
[c_matrix]= confusionmat(actual,label);
[acc_RF,metrics_RF]=Weka_statistics(c_matrix,[classname])

% %NaiveBayes
% SVMModel = fitcecoc(x,y,'Learners', 'naivebayes');
% [label, score] = predict(SVMModel,ML_TableTest{:,1:end-1});%Test
% actual=ML_TableTest{:,end};
% label=string(label);
% [c_matrix]= confusionmat(actual,label);
% acc_naive=Weka_statistics(c_matrix,[classname])
% 
% 
 end
end


% disp(' ')
%  disp('%-------- CV Training Dataset Accuracy -----------%')
% fprintf('ABayes 2fold                        %0.2f %%\n',acc2f(LP)*100);
% % fprintf('LDA 2fold                         %0.2f %%\n',(1-LDArate2(LP))*100);
% fprintf('SVM 2fold                           %0.2f %%\n',(1-SVMrate2(LP))*100);
% disp(' ')
% fprintf('ABayes 10fold                     %0.2f %%\n',acc10f(LP)*100);
% % fprintf('LDA 10fold                       %0.2f %%\n',(1-LDArate10(LP))*100);
% fprintf('SVM 10fold                         %0.2f %%\n',(1-SVMrate10(LP))*100);
% disp('')
% fprintf('ABayes holdout                     %0.2f %%\n',acc_ABayes*100);
%  
%% Compile participant data into one array for holdout
PR_completed(participant,1)=ParID;
Par_Abayes_hold(participant,1)=acc_ABayes(LP);
Par_SVM_hold(participant,1)=acc_SVM(LP);
Par_DT_hold(participant,1)=acc_DT(LP);
Par_RF_hold(participant,1)=acc_RF(LP);

Par_Abayes_holdmetrics(participant,:)=metrics_ABayes;
Par_SVM_holdmetrics(participant,:)=metrics_SVM;
Par_DT_holdmetrics(participant,:)=metrics_DT;
Par_RF_holdmetrics(participant,:)=metrics_RF;

% Par_Abayes_2fmetrics(participant,:)=acc2fmetrics;
% Par_Abayes_10fmetrics(participant,:)=acc10fmetrics;
% Par_SVM_2fmetrics(participant,:)=SVMrate2metrics;
% Par_SVM_10fmetrics(participant,:)=SVMrate10metrics;
% Par_DT_2fmetrics(participant,:)=DTrate2metrics;
% Par_DT_10fmetrics(participant,:)=DTrate10metrics;
% Par_RF_2fmetrics(participant,:)=RFrate2metrics;
% Par_RF_10fmetrics(participant,:)=RFrate10metrics;

%Get SFS features for each participant
% Par_SFSfeatures(participant,1)={newF'};
Par_SFSnamesAB(participant,1)=FS_namesAB;
Par_SFSnamesSVM(participant,1)=FS_namesSVM;
Par_SFSnamesDT(participant,1)=FS_namesDT;
Par_SFSnamesRF(participant,1)=FS_namesRF;
    catch
    end
end

%combine arrays for excel
Par_excel=[PR_completed,Par_Abayes_hold,Par_SVM_hold,Par_DT_hold,Par_RF_hold];
Par_excelm=[PR_completed,Par_Abayes_holdmetrics,Par_SVM_holdmetrics,Par_DT_holdmetrics,Par_RF_holdmetrics];
Task=[string(repmat('Nback',size(PR_completed,1)*4,1))];
win=[repmat(Epochsize,size(PR_completed,1)*4,1)];
typtest= [string(repmat('Holdout',size(PR_completed,1)*4,1))];
ML_names=[string(repmat('ABayes',size(PR_completed,1),1)); string(repmat('SVM',size(PR_completed,1),1)); string(repmat('DT',size(PR_completed,1),1)); string(repmat('RF',size(PR_completed,1),1))];
parcom=[repmat(PR_completed,4,1)];
ML_metrics= [Par_Abayes_holdmetrics;Par_SVM_holdmetrics;Par_DT_holdmetrics;Par_RF_holdmetrics];
Excelinput=table(Task,win,typtest,ML_names,parcom,ML_metrics);
% Par_excel2m=[PR_completed,Par_Abayes_2fmetrics,Par_SVM_2fmetrics,Par_DT_2fmetrics,Par_RF_2fmetrics];
% Par_excel10m=[PR_completed,Par_Abayes_10fmetrics,Par_SVM_10fmetrics,Par_DT_10fmetrics,Par_RF_10fmetrics];

featHO=Par_SFSnamesAB;
unpackf10_2=padcat(featHO{:});
Par_com10=num2cell(PR_completed);
% Par_com10(cell2mat(Par_com10) == 0) = [];
parrep10=repmat(Par_com10, 1,1);
Task10=[string(repmat('Nback',size(parrep10,1),1))];
win10=[repmat(Epochsize,size(parrep10,1),1)];
typtest10= [string(repmat('Holdout',size(parrep10,1),1))];
%Ouput for EXCEL
pfeatHO=[Task10,typtest10,win10, parrep10,unpackf10_2];

%For rearranging the feature list into a format for Rstudio to read
feat_table=table(pfeatHO);
feat_table = splitvars(feat_table);
feat_table.Properties.VariableNames={'Task','Analysis','Window','ID','1','2','3','4','5','6','7','8','9','10','11','12'} %'var12','var13','var14','var15','var16'
pfeatHO_long = stack(feat_table,5:16,...
            'newDataVariableName','feature',...
          'IndexVariableName','Indx');
        

%enable plots again
set(gcf,'Visible','on');              
set(0,'DefaultFigureVisible','on');

%get the list of choosen FS from SFS (this works, just havent rewritten it to aggreaget for the differen ML algorthims)
% ListnamesFS=[Par_SFSnames{:}];
% [ListnamesFSUn,~,idx] = unique(ListnamesFS,'stable');
% [ListnamesFSUnSort,ListIdxSortbycount]=sort(groupcounts(idx),'descend');

% figure;
% bar(1:length(ListIdxSortbycount),ListnamesFSUnSort);
% grid on
% title('SFS - most selected features for participants')
% xlabel('Features selected by SFS')
% ylabel('Number of Paricipants using feature')
% xticks(1:length(ListnamesFSUn))
% xticklabels(strrep(ListnamesFSUn(ListIdxSortbycount),'_','\_'))
% xtickangle(45)

%%
function [Num]=Label2Num(labels)
Num=zeros([length(labels),1]);                              
Num(find(strcmp(labels,'L')))=1;
Num(find(strcmp(labels,'M')))=2;
Num(find(strcmp(labels,'H')))=3;
end