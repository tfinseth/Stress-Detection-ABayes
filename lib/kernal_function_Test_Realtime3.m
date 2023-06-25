function [accout,nDVsout,framepred,metrics,test_col_class_conmax,test_col_con, c_matrix] = kernal_function_Test_Realtime3(testfile,table,list,classnametestvar)
%%Kernal Distribution classification
%MULTIPLE BIOMETERIC DV - TrainTest function for Realtime
%
%Author: Tor Finseth
%Date last edited: 7/16/2021
%
%
%NOTE: ONLY WORKS FOR 3 CLASSES
%FileType: .csv, .arff, or matlab table

%INPUT: 
%   'testfile' is a table of the testing dataset
%   'table' is a table of the training dataset
%   'list' is cellarray of feature names
%   'classnametest' is a string array of the order of occurance of unique
%   classes in the testing, Needed when labels are '?' for CM and accuracy
%   calculations, otherwise the CM will have values in the wrong cells.

%VERSION: This was created to for fixing the SFS with
%Kfold_FeatiresSFS_onML_Table_vCV_v2.m specifically it has been changed to
%output the CM so I can average multiple folds in the parent program.



margin= {'','','n'}; % 3rd variable..... 'y'=equal class weights of 33%, 'n'=weights based on class size


%% Read in TESTING file
% this is the testing table created in realtime code
%filename_test = "U:/SPACE_ASTRONAUT/PHD/Experiments/6 - EXP#2_Stress Gauge/1 - DATA/Biopac_data/P6/P6_4-3-19_MATB_shortenedfortest.csv";
%filename_test = "U:/SPACE_ASTRONAUT/PHD/Experiments/6 - EXP#2_Stress Gauge/1 - DATA/Biopac_data/P12/P12_4-10-19_Matb_50_TEST.csv";
if istable(testfile)==0
    filename_test = testfile;
    [filepath,name,ext] = fileparts(filename_test);
    ext_test=string(ext);
else
    ext_test=string('table');
end
%% CONVERT data from testing file
%Convert to table, depending on filetype: .arff or .csv 
if ext_test == '.arff'
addpath('U:\SPACE_ASTRONAUT\PHD\Experiments\6 - EXP#2_Stress Gauge\wekalab-master-copy')
D = wekaLoadData(filename_test);
[data,attributes,classIndex,stringVals,relationName] = weka2matlab(D,{});
%throw error if no class column [add code here]
attstring=string(attributes);
table_test=array2table(data,'VariableNames',attstring);
list_test = strrep(attstring,'class','');
list_test = cellstr(list_test);
classnametest=unique(table_test.class);
    if string(classnametest) == '?'
        confusiontoggle = 0;
    else 
        confusiontoggle = 1;
    end
    for i=1:classIndex-1
    DV=strcat('table.',attributes(:,i));
    DV=DV{1};
    table_test.(i)=cell2mat(eval(DV));
    end
list_test(cellfun('isempty',list_test)) = [];
end

if ext_test == '.csv'
table_test = readtable(filename_test);
classnametest=unique(table_test.class,'stable');
    if string(classnametest) == '?'
        confusiontoggle = 0;
    else 
        confusiontoggle = 1;
    end
list_test = table_test.Properties.VariableNames;
%throw error if no class column [add code here]
list_test = strrep(list_test,'class','');
list_test(cellfun('isempty',list_test)) = []; 
end

if istable(testfile)
    table_test=testfile;
    classnametest=unique(table_test.class,'stable');
    classnametrain=unique(table.class,'stable');
    if string(classnametest) == '?'
        confusiontoggle = 0;
        classnametest=classnametestvar;
        classname=classnametrain;
    else 
        confusiontoggle = 1;
        classname=classnametrain;
    end
list_test = table_test.Properties.VariableNames;
%throw error if no class column [add code here]
list_test = strrep(list_test,'class','');
list_test(cellfun('isempty',list_test)) = []; 
end

%throw error if wrong filetype
if (ext_test ~= '.csv') && (ext_test ~= '.arff') && (istable(testfile)==0)
    disp ('ERROR: Filetype not supported. File should be a .csv or .arff.')
end
%% Get user input, weights, DVs, scatter plots
%{
[indx] = listdlg('PromptString','Select measures:',...
                           'SelectionMode','multiple',...
                           'ListString',list);
prompt = {'Enter weight, Class 1:', 'Enter weight, Class 2:', 'Enter weight, Class 3:', 'Scatter plots? (y/n)'};
title2 = 'Input';
dims = [1 1 1 1];
definput = {'0.33', '0.33', '0.33','n'};
answer = inputdlg(prompt,title2,dims,definput);
%}
if istable(testfile)
    DVs=[1:size(testfile,2)-1];
else
    DVs=[1,2,3,4,5,6,7,8];   
end
B=DVs(DVs~=0);
indx = B;
[ncolDV,nDVsout]=size(B);
answer = {'0.33', '0.33', '0.33','n'};
scat=cell2mat(answer(4));
%disp('Input:')
%list(indx)
[ro,col]=size(indx);
%% standardize the training table DVs, if more than one DV
if col>1
oldtable_train=table;
oldtable_test=table_test;
for cl=1:col
DVselected=strcat('table.',list(indx(cl)));
DVselected=DVselected{1};

DVselected_test=strcat('table_test.',list(indx(cl)));
DVselected_test=DVselected_test{1};

%TRAIN - DV signal values, seperated by class, used in next portion of code
DV_train = eval(DVselected); %class 1
stdDV_train =(DV_train-mean(DV_train))/std(DV_train);
table(:,indx(cl))=num2cell(stdDV_train);

%TEST - standardize the testing table DVs based on the training table (M,SD)
DV_test = eval(DVselected_test); %class 1
stdDV_test =(DV_test-mean(DV_train))/std(DV_train);
table_test(:,indx(cl))=num2cell(stdDV_test);

end
end
%{
%% standardize the testing table DVs based on the training table (M,SD)
if col>1
oldtable_test=table_test;
for cl=1:col
DVselected_test=strcat('table_test.',list(indx(cl)));
DVselected_test=DVselected_test{1};

%DV signal values, seperated by class, used in next portion of code
DV_test = eval(DVselected_test); %class 1
stdDV_test =(DV_test-mean(DV_train))/std(DV_train);
table_test(:,indx(cl))=num2cell(stdDV_test);
end
end
%}

%% Seperate by Class
class = categorical(table.class);
q='''';

%get start a loop to look at different DVs
fignum=1;
for cl=1:col
DVselected1=strcat('table.',list(indx(cl)),'(class==',q,classname(1),q,')');
DVselected2=strcat('table.',list(indx(cl)),'(class==',q,classname(2),q,')');
DVselected3=strcat('table.',list(indx(cl)),'(class==',q,classname(3),q,')');
DVselected1=DVselected1{1};
DVselected2=DVselected2{1};
DVselected3=DVselected3{1};

%DV signal values, seperated by class, used in next portion of code
Class_1 = eval(DVselected1); %class 1
Class_2 = eval(DVselected2); %class 2
Class_3 = eval(DVselected3); %class 3

%seperate the DV-signal values to plot against another DV
C_1(:,cl) = Class_1; %class 1
C_2(:,cl) = Class_2; %class 2
C_3(:,cl) = Class_3; %class 3

Idx_1= classname(1);
Idx_2= classname(2);
Idx_3= classname(3);
Idx_1= string(Idx_1);
Idx_2= string(Idx_2);
Idx_3= string(Idx_3);


%% Fit Kernal distributions to grouped sample data, g_k, where k=3 classes
%https://www.mathworks.com/help/stats/ksdensity.html
%Kernel smoothing function estimate for univariate and bivariate data
%pd = fitdist(MPG,'Kernel','BandWidth',4)
[fi2,xi] = ksdensity(Class_1);
[fj2,xj] = ksdensity(Class_2);
[fk2,xk] = ksdensity(Class_3);
%find x range that fits all three classes for ONE DV(i)
minValue(cl) = min([xi(:);xj(:);xk(:)]);
maxValue(cl) = max([xi(:);xj(:);xk(:)]);
end
%determine the min and max x-axis values between the DVs
mminValue = min(minValue);
mmaxValue = max(maxValue);
x=linspace(mminValue,mmaxValue,100);
x=x';
x=repmat(x,1,col);

%% Add probability weights to each class
%must sum to 1
%graph above and below will have matching form when weights are equal (but
%different values)
if strcmp(margin(3),'y')
    pi_1=str2num(cell2mat(answer(1)));
    pi_2=str2num(cell2mat(answer(2)));
    pi_3=str2num(cell2mat(answer(3)));
else
    pi_1=length(C_1)/(length(C_1)+length(C_2)+length(C_3)); %weights from training folds
    pi_2=length(C_2)/(length(C_1)+length(C_2)+length(C_3));
    pi_3=length(C_3)/(length(C_1)+length(C_2)+length(C_3));
    %pi_1=total_size(1); %weights from training dataset (not folds)
    %pi_2=total_size(2);
    %pi_3=total_size(3);
end

%weight check
if pi_1+pi_2+pi_3<0.99
    disp('ERROR: WEIGHTS DO NOT SUM TO ONE')
end
if pi_1+pi_2+pi_3>1
    disp('ERROR: WEIGHTS SUM IS GREATER THAN ONE')
end

%determine bandwidth
%bw_1=0.2
%bw_2=0.2
%bw_3=0.2
%Silvermans rule for bandwidth (std is already 1 if standardized in
%beginning), I added the *2 to smooth even more
bw_1=std(C_1(:))*(4/((col+2)*length(C_1)))^(1/(col+4))*2;
bw_2=std(C_2(:))*(4/((col+2)*length(C_2)))^(1/(col+4))*2;
bw_3=std(C_3(:))*(4/((col+2)*length(C_3)))^(1/(col+4))*2;

fi=pi_1*mvksdensity(C_1,x,'Bandwidth',bw_1);
fj=pi_2*mvksdensity(C_2,x,'Bandwidth',bw_2);
fk=pi_3*mvksdensity(C_3,x,'Bandwidth',bw_3);

%% estimate density for testing dataset values

    %group the DV values into one array per row, run as vector
    x_sp=table_test{:,indx};

    %find density for chosen value, multiply weights
    %vectorized is faster than for loop, but parrellel is 3 times faster
    %[fi_sp] = pi_1.*mvksdensity2(C_1,x_sp,'Bandwidth',bw_1);
    %[fj_sp] = pi_2.*mvksdensity2(C_2,x_sp,'Bandwidth',bw_2);
    %[fk_sp] = pi_3.*mvksdensity2(C_3,x_sp,'Bandwidth',bw_3);
    Input_w = [pi_1;pi_2;pi_3]; 
    Input_bw = [bw_1;bw_2;bw_3];
    Input_C = {C_1;C_2;C_3};
    % Perform all 3 calculations together
    % If testing data doesnt match training data, Results will be zeros


    for i=1:3
        [Results{i}] = Input_w(i).*mvksdensity(Input_C{i,1},x_sp,'Bandwidth',Input_bw(i));
    end
    %Results = [Results{:}];

    % Find conditional probability from input already weighted
    f= sum([Results{1,1},Results{1,2},Results{1,3}],2);
    Pi_sp=Results{1,1}./f;
    Pj_sp=Results{1,2}./f;
    Pk_sp=Results{1,3}./f;

    %Find max conditional probabilties for each class
    [test_col_class_conmax,test_col_class] = max([Pi_sp,Pj_sp,Pk_sp],[],2); %M is the largest probability, I is the class incidies

    %Conditional probabilties for each class
    test_col_con=[Pi_sp,Pj_sp,Pk_sp];
    
%% Calculate Confusion Matrix 
classcol=col+1;
classarray=table2array(table(:,classcol));
%[uv2,~,actual] = unique(table_test.class,'stable'); % fixed 5/10/19. giving actual class wrong indx.
actual=zeros(length(table_test.class),1);
for i=1:length(table_test.class)
    if table_test.class(i) == string(classnametrain(1))
        actual(i)=1;
    elseif table_test.class(i) == string(classnametrain(2))
        actual(i)=2;
    elseif table_test.class(i) == string(classnametrain(3))
        actual(i)=3;
    end
end
%class = categorical(table_test.class);
%actual=grp2idx(class);
actual=num2str(actual);
actual=cellstr(actual);
predict=test_col_class;
predict=num2str(predict);
predict=cellstr(predict);
%This ouputs predictions, if the class is '?'
if confusiontoggle == 0
    %[accout,framepred,CMR] = ConfusionSetupRT(predict,classname); %similar to the code below, except for RT
    %CMN = [classname'; CMR];
    %disp(CMN)
    [framepred]=predict;
        framepred(cellfun(@(elem) elem == '1', framepred)) = {classname(1)};
        framepred(cellfun(@(elem) elem == '2', framepred)) = {classname(2)};
        framepred(cellfun(@(elem) elem == '3', framepred)) = {classname(3)};
    actual=ones(length(predict),1); %actual is 1, because the prediction display is made from the top row of confusion matrix
    actual=num2str(actual);
    actual=cellstr(actual);
    [accout] = 'NaN';
    %[c_matrix]= confusionmat2(actual,predict,'order',{'L' 'M' 'H'}); % Edited confusion file because doesnt give matrix for a 100% perfect score with only one class present (e.g., all 'L')
    [c_matrix]= confusionmat(actual,predict,'order',{'1' '2' '3'}); %Noted in Datalog_realtime, the confusionmat2 was messing up the prediction display   
        CM = num2cell(c_matrix);
        CMR=CM(1,1:3); %take top row of CM, number items in each class
        CMN = [classnametest'; CMR]; 
        %disp(CMN);
        [~,MaxClass]=max(cell2mat(CMR));
        metrics='NA';
end

%This outputs confusion matrix, if classes are given
if confusiontoggle == 1
    [framepred]=predict;
    framepred(cellfun(@(elem) elem == '1', framepred)) = {classname(1)};
    framepred(cellfun(@(elem) elem == '2', framepred)) = {classname(2)};
    framepred(cellfun(@(elem) elem == '3', framepred)) = {classname(3)};
    [frameactual]=actual;
    frameactual(cellfun(@(elem) elem == '1', frameactual)) = {classname(1)};
    frameactual(cellfun(@(elem) elem == '2', frameactual)) = {classname(2)};
    frameactual(cellfun(@(elem) elem == '3', frameactual)) = {classname(3)};
    %[c_matrix]= confusionmat2(frameactual,framepred,'Order',{'L' 'M' 'H'}); % Edited confusion file because doesnt give matrix for a 100% perfect score with only one class present (e.g., all 'L')
    [c_matrix]= confusionmat(actual,predict,'Order',{'1' '2' '3'}); %Noted in Datalog_realtime, the confusionmat2 was messing up the prediction display
        [accout,metrics]=Weka_statistics(c_matrix,classname)
        CM = num2cell(c_matrix);
        CMN = [classname'; CM(1,1:3)]; %want in terms of training LMH
        disp(CMN)
        [~,MaxClass]=max(cell2mat(CM(1:3)));
end
%{
%% ----------------COMMENTED OUT ALL THE GRAPHS -----------------------
%% Plot the density distributions with weights
figure(fignum)
fignum=fignum+1;
plot(x(:,1),fi,'b')
hold on
plot(x(:,1),fj,'g')
plot(x(:,1),fk,'r')

%{ 
%to test that the claculated x range matches the original distribution
plot(xi,pi_1*fi2,'b.')
plot(xj,pi_2*fj2,'g.')
plot(xk,pi_3*fk2,'r.')
%}

%plot the input value
legend(Idx_1,Idx_2,Idx_3)

if col>1
runline=strcat('Probabiltiy Density Distribution, multivariate z-score, by stress class');
else
runline=strcat('Probabiltiy Density Distribution,', list(indx),',by stress class');
end
title(runline)
xlabel(list(indx))
hold off
%% Plot conditional probabilities
%{
%find x range
minValue = min([xi(:);xj(:);xk(:)]);
maxValue = max([xi(:);xj(:);xk(:)]);
x=linspace(minValue,maxValue,100);

%find conditional probability given x
for i=1:100
f(i)=pi_1*mvksdensity(Class_1,x(i)) + pi_2*mvksdensity(Class_2,x(i)) + pi_3*mvksdensity(Class_3,x(i));
Pi(i)=pi_1*mvksdensity(Class_1,x(i))/f(i);
Pj(i)=pi_2*mvksdensity(Class_2,x(i))/f(i);
Pk(i)=pi_3*mvksdensity(Class_3,x(i))/f(i);
end
%}

for i=1:100
f(i)=pi_1*mvksdensity(C_1,x(i,:),'Bandwidth',bw_1) + pi_2*mvksdensity(C_2,x(i,:),'Bandwidth',bw_2) + pi_3*mvksdensity(C_3,x(i,:),'Bandwidth',bw_3);
Pi(i)=pi_1*mvksdensity(C_1,x(i,:),'Bandwidth',bw_1)/f(i);
Pj(i)=pi_2*mvksdensity(C_2,x(i,:),'Bandwidth',bw_2)/f(i);
Pk(i)=pi_3*mvksdensity(C_3,x(i,:),'Bandwidth',bw_3)/f(i);
end

figure(fignum)
fignum=fignum+1;
plot(x(:,1),Pi,'b')
hold on
plot(x(:,1),Pj,'g')
plot(x(:,1),Pk,'r')


%plot the input value
legend(Idx_1,Idx_2,Idx_3)

if col>1
runline=strcat('Conditional Probability, multivariate z-score, by stress class');
else
runline=strcat('Conditional Probability,', list(indx),',by stress class');
end
title(runline)
xlabel(list(indx))
ylabel('P(class|x)')
hold off

%% plot DV vs DV 
%only works if multiple DVs are selected
if scat ~= 'n'
if col>1
    % Graph settings
    if col>2
        for k=1:col-1
        figure(fignum)
        fignum=fignum+1;
            scatter(C_1(:,k),C_1(:,k+1),'b')
            hold on
            scatter(C_2(:,k),C_2(:,k+1),'g')
            scatter(C_3(:,k),C_3(:,k+1),'r')
            xlabel(list(indx(k)))
            ylabel(list(indx(k+1)))
            legend(Idx_1,Idx_2,Idx_3)
        end
    end
        figure(fignum)
        fignum=fignum+1;
            scatter(C_1(:,1),C_1(:,col),'b')
            hold on
            scatter(C_2(:,1),C_2(:,col),'g')
            scatter(C_3(:,1),C_3(:,col),'r')
            xlabel(list(indx(1)))
            ylabel(list(indx(col)))
            legend(Idx_1,Idx_2,Idx_3)
end
end

%% ROC curve
%updated 5-19-2019
%Needs statistics Matlab Addon

diffscore1 = test_col_con(:,1) - max(test_col_con(:,2),test_col_con(:,3));
diffscore2 = test_col_con(:,2) - max(test_col_con(:,1),test_col_con(:,3));
diffscore3 = test_col_con(:,3) - max(test_col_con(:,1),test_col_con(:,2));
[X1,Y1,T,~,OPTROCPT,suby,subnames] = perfcurve(table_test.class,diffscore1,Idx_1);
[X2,Y2,T,~,OPTROCPT,suby,subnames] = perfcurve(table_test.class,diffscore2,Idx_2);
[X3,Y3,T,~,OPTROCPT,suby,subnames] = perfcurve(table_test.class,diffscore3,Idx_3);
plot(X1,Y1,'r',X2,Y2,'b',X3,Y3,'g')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Approx. Bayes Classification of Stress Levels')
x4 = [0 1];
y4 = [0 1];
ln=line(x4,y4,'Color','black');
set(get(get(ln,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
legend(Idx_1,Idx_2,Idx_3)
%}