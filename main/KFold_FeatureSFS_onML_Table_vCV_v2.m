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
addpath('...\lib\hhentschke-measures-of-effect-size-toolbox-3d90ae5')
%Where the Pans-Tompkins files are located
addpath('...\lib')
%Weka_statistics
addpath('...\lib')
%features
addpath('...\lib\catch22-master\wrap_Matlab')

%Select the participants you want to run (note: larger window sizes cause some participants to error. Below are suggested participant combinations. Change 'Epochsize' to match selection)
%Note: Id=12 removed, becuse not all data recorded for ISS-High
% VR-ISS ----------
% Participant_arr= [66,64,63,62,61,57,56,55,54,52,51,50,49,47,45,43,42,41,31,29,28,27,26,24,23,22,19,18,17,16,15,14,13,11,9,8,5];     % ISS original, removed 42,12,29 for 20sec, 49,15 for 30sec, 21 for 40 sec, 47 for 30-60sec, Holdout 56,54,43,34 for 20,30sec (NF), 27,19 for 30sec, 66,57,54,52,45,41,18 for 40secc ---
% Participant_arr= [66,64,63,62,61,57,56,55,54,52,51,50,49,47,45,43, 41,28,24,23,22,19,18,16,15,13,9];     % ISS 10-sec,
% Participant_arr= [66,64,63,62,61,57,56,55,54,52,51,50,49,47,45,43, 41,28,24,23,22,19,18,16,15,13,9];     % ISS 20-sec,  --- deleted 
% Participant_arr= [66,64,63,62,61,57,56,55,54,52,51,50,49,47,45,43, 41,28,24,23,22,19,18,16,15,13,9];     % ISS 30-sec,  
% Participant_arr= [66,64,63,62,61,57,56,55,54,52,51,50,49,47,45,43, 41,28,24,23,22,19,18,16,15,13,9];     % ISS 40-sec, 62, 51,50, 28,24,22, deleted  
% Nback ----------
% Participant_arr= [68,67,39,38,37,36,29,27,26,17,14,11,8,5]; % Nback 10 seconds
% Participant_arr= [68,67,39,38,37,36,29,27,26,17,14,11,8,5]; % Nback 20 seconds
% Participant_arr= [39,38,37,36,29,27,26,17,14,11,8,5]; % Nback 30 seconds
Participant_arr= [68,67,39,38,37,36,29,27,26,17,14,11,8,5]; % Nback 40 seconds

TT=2;% Holdout=2, but adapted this to run as a CV with multiple HO


%SFS inputs
hout=.2; %ratio of holdout for SFS crit
SFSfold=10; %number of folds in SFS feature selection
iteration=1; %number of iterations that SFS should make, feauters will then be selected by subset unions.

%Run 2fold an 10 fold
for CVval=2:2
    if CVval==1
        CVfold=2;
    else
        CVfold=10;
    end
for participant=1:length(Participant_arr)
    ParID=Participant_arr(participant);
    try %if it errors because not enought testing data, then just skip  the participant
%% INPUTS
Epochsize=40; %In seconds, what the length of epochs should be.
sample_fre=125; %Hz
Trim_StartLen=0; %sec, trim from beginning of stress trial
Trim_EndLen=0; %sec, trim from beginning of stress trial
Trimtest=0; %Also trim the testing data for validation, 1=True, 0 =False
PlotFreAnalaysis=0; %Plot FFT,Mel-Spectrogram, Scalogram for ECG and NIBP 1=true, 0=false

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

    close all
    filename=filename_arr(1)
    class_times=class_times_arr(1,:);
    class_times=[class_times{:}];
    class_order=class_order_arr(1,:)'; 
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

    T_train=T;
    classes_train=classes;

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
    [tsffeat{j,i},tsffeat_names]=tsfreshMATLAB_welch(y,sample_fre);

    if i==1 %Only calculate once, not for every DV column
        [HRVfeatures{j,i},HRVfeatures_names] = GetHRVfeatures(j,T,sample_fre,firstdpt,lastdpt,ECG_col,NIBP_col,PlotFreAnalaysis);
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

%% Get the CV partitions

%Prep data for holdout
[~,n]=size(ML_Table);
Ctable = table2cell(ML_Table);
class_label=[Ctable{:,n}]';

CV2F = cvpartition(class_label,'KFold',CVfold); %randomized, stratified partitions
    for i = 1:CV2F.NumTestSets
        trIdx = CV2F.training(i);
        teIdx = CV2F.test(i);
        Train{i}=Ctable(trIdx,:);
        Test{i}=Ctable(teIdx,:);
        Label{i}=class_label(trIdx,:);
    end
%Now that I have the partisiions for the trainign and testing, I want to
%loop the feature selection and classficiation for the different dataset
%partitions.
for pari = 1:CV2F.NumTestSets
 %clean up from last partition
 if pari>1
     clearvars ML_Table Std_ML_Table mu_train sigma_train
 end
 
%get the trainign dataset fo this partition
ML_Table=cell2table(Train{pari});
ML_Table.Properties.VariableNames = Vnames;

%% Classify, & Compile Fstat and FeatureID in array
    %Standardize the ML_table wrt. itself

    for i=1:size(ML_Table,2)-1
        [Std_ML_Table{i} mu_train(i) sigma_train(i)]= zscore(ML_Table{:,i}); %Standardize ML_table wrt. itself 
    end
    table_class=ML_Table(:,end);
    Std_ML_Table=table(Std_ML_Table{1:size(ML_Table,2)-1});
    Std_ML_Table.Properties.VariableNames = ML_Table.Properties.VariableNames(1:size(ML_Table,2)-1);
    ML_TableTrain= [Std_ML_Table table_class];
    
%% Run Sequential Feature Selection on the training datasets

addpath('C:\Users\tfinseth\Documents\MATLAB\Add-Ons\Collections\SVM-RFE');

%Prep Traning data for SFS
[~,n]=size(ML_TableTrain);
Ctable2 = table2cell(ML_TableTrain);
Std_dataTrain = cell2mat(Ctable2(:,1:n-1));
feature_VariableNames=ML_TableTrain.Properties.VariableNames(1:n-1);
class_labels=[Ctable2{:,n}]';

%Prep unstandardized Testing partition
ML_TableTest=cell2table(Test{pari});
ML_TableTest.Properties.VariableNames = Vnames;
ML_Table_unstd=ML_TableTest; % this table is unstandardized

%standardize TableTest wrt TableTrain
for i=1:size(ML_Table_unstd,2)-1
    ML_TableTest{:,i}=(ML_Table_unstd{:,i}-mu_train(i))/sigma_train(i);
end
%Prep Test data for later
[~,n]=size(ML_TableTest);
Ctable2 = table2cell(ML_TableTest);
Std_dataTest = cell2mat(Ctable2(:,1:n-1));
class_labels2=[Ctable2{:,n}]';

% Perform SFS, sequentialfs creates candidate feature subsets by sequentially adding each of the features not yet selected.
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
            [newF] = IterativeSequentialfs_v2(Std_dataTrain,class_labels,hout,SFSfold,iteration,algor)
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
            FS_namesAB{pari}=feature_VariableNames(newF); %names features selected
        end
        if algor==2 %SVM
            ML_TableTrainSVM = addvars(temp1,class_labels,'after',size(temp1,2));
            ML_TableTrainSVM.Properties.VariableNames{'class_labels'} = 'class';
            ML_TableTestSVM = addvars(temp2,class_labels2,'after',size(temp2,2));
            ML_TableTestSVM.Properties.VariableNames{'class_labels2'} = 'class';
            FS_namesSVM{pari}=feature_VariableNames(newF); %names features selected
        end
        if algor==3 %DT
            ML_TableTrainDT = addvars(temp1,class_labels,'after',size(temp1,2));
            ML_TableTrainDT.Properties.VariableNames{'class_labels'} = 'class';
            ML_TableTestDT = addvars(temp2,class_labels2,'after',size(temp2,2));
            ML_TableTestDT.Properties.VariableNames{'class_labels2'} = 'class';
            FS_namesDT{pari}=feature_VariableNames(newF); %names features selected
        end
        if algor==4 % RF
            ML_TableTrainRF = addvars(temp1,class_labels,'after',size(temp1,2));
            ML_TableTrainRF.Properties.VariableNames{'class_labels'} = 'class';
            ML_TableTestRF = addvars(temp2,class_labels2,'after',size(temp2,2));
            ML_TableTestRF.Properties.VariableNames{'class_labels2'} = 'class';
            FS_namesRF{pari}=feature_VariableNames(newF); %names features selected
        end
end
%% Classify - get the CM for each fold then aggregate and average, then compute perfromance metrics for each alogthirm
FS_names={feature_VariableNames(newF)}; %names features selected

set(gcf,'Visible','off');  %supress plots            
set(0,'DefaultFigureVisible','off');

%ground truth
 actual=ML_TableTestSVM{:,end};
        
 %ABayes
 features=ML_TableTestAB.Properties.VariableNames(1:end-1);
 [classname,~,~] = unique(ML_TableTrainAB.class,'stable');
 [~,~,~,~,~,~, c_matrix_AB{pari}] = kernal_function_Test_Realtime3(ML_TableTestAB,ML_TableTrainAB,features);

%https://www.mathworks.com/help/stats/fitcecoc.html, 'Learners'
%SVM
y=ML_TableTrainSVM{:,end};
x=ML_TableTrainSVM{:,1:end-1};
SVMModel = fitcecoc(x,y,'Learners', 'svm');
[label, score] = predict(SVMModel,ML_TableTestSVM{:,1:end-1});%Test
actual=ML_TableTestSVM{:,end};
label=string(label);
[c_matrix_SVM{pari}]= confusionmat(actual,label,'Order',{'L' 'M' 'H'});

%Tree
y=ML_TableTrainDT{:,end};
x=ML_TableTrainDT{:,1:end-1};
SVMModel = fitcecoc(x,y,'Learners', 'tree');
[label, score] = predict(SVMModel,ML_TableTestDT{:,1:end-1});%Test
actual=ML_TableTestDT{:,end};
label=string(label);
[c_matrix_DT{pari}]= confusionmat(actual,label,'Order',{'L' 'M' 'H'});

%Random Forest
y=ML_TableTrainRF{:,end};
x=ML_TableTrainRF{:,1:end-1};
RFModel = fitcensemble(x,y,'Method','Bag');
[label, score] = predict(RFModel,ML_TableTestRF{:,1:end-1});%Test
actual=ML_TableTestRF{:,end};
label=string(label);
[c_matrix_RF{pari}]= confusionmat(actual,label,'Order',{'L' 'M' 'H'});


end %end fold testing

%% Aggregate the fold CMs into one matrix to get the perfromance metrics

c_matrix = AggCM(c_matrix_AB, classname);
[~,metrics_ABayes]=Weka_statistics(c_matrix,[classname]);
c_matrix = AggCM(c_matrix_SVM,classname);
[~,metrics_SVM]=Weka_statistics(c_matrix,[classname]);
c_matrix = AggCM(c_matrix_DT,classname);
[~,metrics_DT]=Weka_statistics(c_matrix,[classname]);
c_matrix = AggCM(c_matrix_RF,classname);
[~,metrics_RF]=Weka_statistics(c_matrix,[classname]);

%% Compile averaged metrics into one array with multiple participants
PR_completed(participant,1)=ParID;

Par_Abayes_metrics(participant,:)=metrics_ABayes;
Par_SVM_metrics(participant,:)=metrics_SVM;
Par_DT_metrics(participant,:)=metrics_DT;
Par_RF_metrics(participant,:)=metrics_RF;

%Get SFS features for each participant
Par_SFSnamesAB{participant,1}=FS_namesAB;
Par_SFSnamesSVM{participant,1}=FS_namesSVM;
Par_SFSnamesDT{participant,1}=FS_namesDT;
Par_SFSnamesRF{participant,1}=FS_namesRF;

    catch
    end
end
Par_SFSCV_AB{CVval}=Par_SFSnamesAB;
Par_SFSCV_SVM{CVval}=Par_SFSnamesAB;
Par_SFSCV_DT{CVval}=Par_SFSnamesAB;
Par_SFSCV_RF{CVval}=Par_SFSnamesAB;
%combine arrays for excel

if CVfold==2
    Par_com2=num2cell(PR_completed);
    Par_excel2m=[Par_Abayes_metrics;Par_SVM_metrics;Par_DT_metrics;Par_RF_metrics];
    Task=[string(repmat('VR-ISS',size(PR_completed,1)*4,1))];
    win=[repmat(Epochsize,size(PR_completed,1)*4,1)];
    typtest= [string(repmat('2Fold',size(PR_completed,1)*4,1))];
    ML_names=[string(repmat('ABayes',size(PR_completed,1),1)); string(repmat('SVM',size(PR_completed,1),1)); string(repmat('DT',size(PR_completed,1),1)); string(repmat('RF',size(PR_completed,1),1))];
    parcom=[repmat(PR_completed,4,1)];
    ML_metrics= [Par_Abayes_metrics;Par_SVM_metrics;Par_DT_metrics;Par_RF_metrics];
    %Ouput for EXCEL
    Excelinput2f=table(Task,win,typtest,ML_names,parcom,ML_metrics);
else
    Par_com10=num2cell(PR_completed);
    Par_excel10m=[Par_Abayes_metrics;Par_SVM_metrics;Par_DT_metrics;Par_RF_metrics];
    Task=[string(repmat('Nback',size(PR_completed,1)*4,1))];
    win=[repmat(Epochsize,size(PR_completed,1)*4,1)];
    typtest= [string(repmat('10Fold',size(PR_completed,1)*4,1))];
    ML_names=[string(repmat('ABayes',size(PR_completed,1),1)); string(repmat('SVM',size(PR_completed,1),1)); string(repmat('DT',size(PR_completed,1),1)); string(repmat('RF',size(PR_completed,1),1))];
    parcom=[repmat(PR_completed,4,1)];
    ML_metrics= [Par_Abayes_metrics;Par_SVM_metrics;Par_DT_metrics;Par_RF_metrics];
    %Ouput for EXCEL
    Excelinput10f=table(Task,win,typtest,ML_names,parcom,ML_metrics);
end
end

%%%%%%%%%%%%%%%% UNPACK ABAYES FEATURES %%%%%%%%%%%%%%%%%%%%%
%to make this work you need to download PADCAT and add this in the code row 146:
% M(M==1) = NaN ; % put the fillers in
% if iscell(X) %check if cell
% M=num2cell(M);
% M(cellfun(@(i) isequal(i, 0), M)) = X;
% else %double array
% M{M==0} = X ;   % put the values in
% end

try 
feat2=Par_SFSCV_AB{1};
unpackf2_1 = vertcat(feat2{:});
unpackf2_2=padcat(unpackf2_1{:});
Par_com2(cell2mat(Par_com2) == 0) = [] ;%delete partipants that data errored on
parrep2=repmat(Par_com2, 2,1); 
Task2=[string(repmat('VR-ISS',size(parrep2,1),1))];
win2=[repmat(Epochsize,size(parrep2,1),1)];
typtest2= [string(repmat('2Fold',size(parrep2,1),1))];
%Ouput for EXCEL
pfeat2=[Task2,typtest2,win2, parrep2,unpackf2_2]; %add parID
catch
end

feat10=Par_SFSCV_AB{2};
unpackf10_1 = vertcat(feat10{:});
unpackf10_2=padcat(unpackf10_1{:});
Par_com10(cell2mat(Par_com10) == 0) = [];
parrep10=repmat(Par_com10, 10,1);
Task10=[string(repmat('Nback',size(parrep10,1),1))];
win10=[repmat(Epochsize,size(parrep10,1),1)];
typtest10= [string(repmat('10Fold',size(parrep10,1),1))];
%Ouput for EXCEL
pfeat10=[Task10,typtest10,win10, parrep10,unpackf10_2];

%For rearranging the feature list into a format for Rstudio to read
feat_table=table(pfeat10);
feat_table = splitvars(feat_table);
feat_table.Properties.VariableNames={'Task','Analysis','Window','ID','1','2','3','4','5','6','7','8','9','10','11'} %'var12','var13','var14','var15','var16'
pfeat10_long = stack(feat_table,5:15,...
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

function Cumlulative_matrix = AggCM(confusionMatrix_arr, classNames)
%confusionMatrix_arr should be a cell array with multple CMs in one row
Cumlulative_matrix=[0,0,0;0,0,0;0,0,0];
%If there was not enough of a class, and the CM is wrong dimensions. Check the CM and fix
for k=1:size(confusionMatrix_arr)
count=0;
confusionMatrix=confusionMatrix_arr{k};
if size(confusionMatrix,2) ~= size(Cumlulative_matrix,2)
    nrow=size(Cumlulative_matrix,2)-size(confusionMatrix,2);%diff in size
    for i=1:size(Cumlulative_matrix,2)
        an=strfind(uv2,classNames(i));
        an=[an{:}];
        if isempty(an)
            missingrow=i;
            count=count+1;
            if count==2
                error("two of the three CM labels are missing")
            end
        end
    end
    if missingrow==3
       confusionMatrix(size(confusionMatrix,2)+nrow,size(confusionMatrix,2)+nrow)= 0;
    end
    if missingrow==2
        %insert middle column
        mcol=[0;0];
        xc=confusionMatrix;
        xc=[xc(:,1),mcol,xc(:,2)];
        %insert middle row
        mrow=[0,0,0];
        xc=[xc(1,:);mrow;xc(2,:)];  
        confusionMatrix=xc;
    end
    if missingrow==1
        %insert first column
        mcol=[0;0];
        xc=confusionMatrix;
        xc=[mcol,xc];
        %insert top row
        mrow=[0,0,0];
        xc=[mrow;xc];  
        confusionMatrix=xc;
    end
end
%add the previous confusion matrixes together for k-fold
Cumlulative_matrix=confusionMatrix+Cumlulative_matrix;
end
end