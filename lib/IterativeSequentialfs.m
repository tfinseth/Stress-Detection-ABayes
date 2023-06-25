function [newF] = IterativeSequentialfs(Std_data,class_labels,hout,fold,iteration,ML_TableTrain)
%ITERATIVESEQUENTIALFS 
% Tor Finseth
% 11-10-2020
%   The sequential feature selection process starts by holding out a
%   portion of data, then running ANOVA to determine highly discriminative
%   features. It then sorts the discriminative feaetures and starts by
%   running single feature ata time until the feature with lowest MCE (by
%   SVM 10-fold) is determined, then runs pairs of features and compares,
%   then runs triplets .... and so on. The number of feautres (selected
%   single, pair, triplet..etc) with the lowest MCE is the final selection.
%   The final selection is tested with the Hold-out data to produce a new
%   MCE. 

%   Since the hold-out is picked at random, the ANOVA, selected features,
%   and MCE can change. Therefore, an iterative process was introduced that
%   runs the hold-out-to-SFS multiple times and selects the winning
%   feature combinations based upon lowest MCE from testing the hold-out.

% INPUTS
% Std_data - standardized data in matrix array, columns are features, rows are observations
% class_labels - one column or labels for all the observations
% hout - scalar of amount of data to hold-out
% fold - number of folds to be used in the SFS
% iterations - number of iterations to conduct the hold-out

%UPDATED
% line 88, remove the most correlated (Added 11-19-2020)
uncorr_feat=1; % Use the SFS with correlated features selected? 1=no (i.e., remove correlated features), 0=yes

%%  Iterative Sequentical Feature Selection
% sequentialfs creates candidate feature subsets by sequentially adding each of the features not yet selected.
for i=1:iteration
%hout=.2; %ratio of holdout
%fold=10; %number of folds in discriminant analysis
%rng(1);% For reproducibility, sets the seed for random number generator to 1
holdoutCVP = cvpartition(class_labels,'holdout',round(hout*length(class_labels))); 
dataTrain = Std_data(training(holdoutCVP,1),:);
grpTrain = class_labels(training(holdoutCVP,1));
%find MCE (misclassification error, i.e., the number of misclassified observations divided by the number of observations)
%t-test
%dataTrainG1 = dataTrain(grp2idx(grpTrain)==1,:);
%dataTrainG2 = dataTrain(grp2idx(grpTrain)==2,:);
%dataTrainG3 = dataTrain(grp2idx(grpTrain)==3,:);
%combo1={dataTrainG1,dataTrainG2,dataTrainG3};
%combo2={dataTrainG2,dataTrainG3,dataTrainG1};
%comb2_lab=[classes(2),classes(3),classes(1)];
% for k=1:length(classes) 
%     [~,p(k,:)] = ttest2(cell2mat(combo1(k)),cell2mat(combo2(k)),'Vartype','unequal'); %t-test
%     figure;
%     [f,x]=ecdf(p(k,:));
%     ecdf(p(k,:));
%     title(strcat('T-TEST:',classes(k),' vs. ', comb2_lab(k), ' features'))
%     xline(0.05,'--r',{'Significance','Limit'});
%     xlabel('P value');
%     ylabel('CDF, % of features with strong discrimination power')
%     numfea(k)=find(x>0.05,1)-1;  %get percent of features below p=0.05
%     perfea(k)=f(numfea(k)); %percent of features below p=0.05 for each class combo
% end
% if min(perfea)<0.1
%     disp('WARNING: One of the classes has a discrimination of <10%. The number of selected features will be small.')
%     disp('Recommend retrying the analysis. Press any key to continue.')
%     pause;
% end
% ComboSelected=3; %LM=1, MH=2, LH=3
% [~,featureIdxSortbyP] = sort(p(ComboSelected,:),2); % sort the features by p-value
%preallocate
p = zeros(size(dataTrain,2),1);
Fstat = zeros(size(dataTrain,2),1);
%One-WAY ANOVA
for k=1:size(dataTrain,2) %loop for each feature
    [p(k,:),tabl] = anova1(dataTrain(:,k),grpTrain,'off'); %to show plots= 'on'
    Fstat(k,:) = tabl{2,5};
end
    figure;
    [f,x]=ecdf(p);
    ecdf(p);
    title('One-Way ANOVA, 3-class, features')
    xline(0.05,'--r',{'Significance','Limit'});
    xlabel('P value');
    ylabel('CDF, % of features with strong discrimination power')
    numfea=find(x>0.05,1);  %get percent of features below p=0.05
    perfea=f(numfea); %percent of features below p=0.05 for each class combo

[pnew,featureIdxSortbyP] = sort(p,1); % Does same thing as sorting F-value [~,featureIdxSortbyF] = sort(Fstat,1,'descend');
[Fnew,featureIdxSortbyF] = sort(Fstat,1,'descend');
%choose features with F-value<65, eliminate features that are toooo
%different (e.g., erronous EDA)
%featureIdxSortbyP=featureIdxSortbyP(Fnew<40);

%%%% Find all features that are significant
numpnew=sum(pnew<=0.05);

%%%% Eliminate corrolated feautres
fs1 = featureIdxSortbyP(1:numpnew); %get feature index for sorted p-values
fs2=fs1;
%of the signifcant features, remove the most correlated (Added 11-19-2020)
corrm = corr(dataTrain(:,fs1));
fea=numpnew;
if uncorr_feat==1 %if flag, remove correleated features
    for co=1:fea-1 %col
        del=find(corrm(co,co+1:end)>.80 | corrm(co,co+1:end)<-.80); %find corr in column GT 90%
        fs1(del+co)=[]; %deletes corr features
        fea=length(fs1);
        corrm = corr(dataTrain(:,fs1));
        if fea<=co
            break
        end
    end
end
if fea<3
    fs1=fs2(1:3); %stops the code from erroring on line 169
    fea=3;
end

%%%% Round the features down if there are too many left
%numf=round(min(perfea)*size(feature_VariableNames,2)); %number of features 
% numf=round(min(perfea)*size(Std_data,2)); %number of features
numf=fea;%number of features
% numf=numf-round(.3*numf); %number of features
if numf>12
   numf=12;
end
fs1=fs1(1:numf,:); %reduce features for SFS
  
%nfs = 1:1:numf; %size(feature_VariableNames,2); %numf needs to below the percent of features below p=0.05
testMCE = zeros(1,numf);
%resubMCE = zeros(1,numf);
%classf = @(xtrain,ytrain,xtest,ytest) ... %criterion used to select features and to determine when to stop
%             sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'diagquadratic'))); %Similar to quadratic discriminate analysis, but with a diagonal covariance matrix estimate (naive Bayes classifiers).
% classf = @(xtrain,ytrain,xtest,ytest)loss(fitcecoc(xtrain,ytrain),xtest,ytest); %SVM function
% classf = @(xtrain,ytrain,xtest,ytest)loss(fitctree(xtrain,ytrain),xtest,ytest); %tree function, takes 7 sec to run
%  classf = @(xtrain,ytrain,xtest,ytest)loss(fitcensemble(xtrain,ytrain,'Method','Bag','NumLearningCycles',10),xtest,ytest); %random forest function, takes 1 min to run
% %resubCVP = cvpartition(length(class_labels),'resubstitution');
% for i = 1:numf 
%    fs = featureIdxSortbyP(1:nfs(i));
%    testMCE(i) = crossval(classf,data(:,fs),class_labels,'partition',holdoutCVP)/holdoutCVP.TestSize;
%    %resubMCE(i) = crossval(classf,data(:,fs),class_labels,'partition',resubCVP)/resubCVP.TestSize;
% end
% figure;
% plot(nfs, testMCE,'o');
% xlabel('Number of Features');
% ylabel('MCE');
% legend({'MCE on the test set'},'location','NW');
% title('Simple Filter Feature Selection Method');

%forward sequential feature selection in a wrapper fashion
tenfoldCVP = cvpartition(grpTrain,'kfold',fold); %10-fold on training dataset, random partition




fprintf('SFS %0.0f running... \n',i);
%Forward sequential selection stops when the first local minimum of the cross-validation MCE is found.
try
 tic
[fsLocal,historyCV] = sequentialfs(classf,dataTrain(:,fs1),grpTrain,'cv',tenfoldCVP,'Nf',numf);
toc
catch
    %If there is too little data to perform 10 fold with data, perform 2-fold instead
    twofoldCVP = cvpartition(grpTrain,'kfold',2);
    [fsLocal,historyCV] = sequentialfs(classf,dataTrain(:,fs1),grpTrain,'cv',twofoldCVP,'Nf',numf);
end
%addpath('U:\SPACE_ASTRONAUT\PHD\Experiments\6 - EXP#2_Stress Gauge\wekalab-master-copy\Book-Software\Chapter4\mfiles')
%[cLbest,maxJ]=SequentialForwardFloatingSelection(dataTrain(:,fs1),grpTrain,classf,numf)
% Crit(i,:)=historyCV.Crit;
% In{i,:}=historyCV.In;
% fs1Sorted_arry(i,:)=fs1';
% end
%[S,W] = SFFS(X1,X2,y1,y2,k,t,N)


%selectedfeaturesLocal=feature_VariableNames(fs1(fsLocal))';
%testMCELocal = crossval(classf,Std_data(:,fs1(fsLocal)),class_labels,'partition', holdoutCVP)/holdoutCVP.TestSize
%[fsCVfor50,historyCV] = sequentialfs(classf,dataTrain(:,fs1),grpTrain,...
%    'cv',tenfoldCVP,'Nf',numf);
figure;
plot(historyCV.Crit,'o');
xlabel('Number of Features');
ylabel('CV MCE');
title('Forward Sequential Feature Selection with cross-validation');
ylim([0 1])

numFSE_vars=find(historyCV.Crit==min(historyCV.Crit),1); %find number of features that make smallest MCE, add ",1" to get the lowest amount of features
if size(numFSE_vars,2)>1 %if multiple combintions have the same MCE
    numFSE_vars=numFSE_vars(end);
end
if (numFSE_vars==1 || numFSE_vars==2)
    numFSE_vars=3;
end
    newF = fs1(historyCV.In(numFSE_vars(end),:));
    [orderlist,ignore] = find( [historyCV.In(1,:); diff(historyCV.In(1:numFSE_vars(end),:) )]' );
    fs1Sorted_arry{i,:}=fs1(orderlist)';
    testMCECVfor10(i) = crossval(classf,Std_data(:,newF),class_labels,'partition',...
        holdoutCVP)/holdoutCVP.TestSize;
end    
% Select final features after iterating FSF
 [~,MCEsorted] = sort(testMCECVfor10,2);%sort the iterations by their lowest crit value
 MCESorted_arry=fs1Sorted_arry(MCEsorted); %sort feature lists by best MCE
%  newF=unique([MCESorted_arry{:}]); %select all features that appear at least twice
%  for j=1:size(newF,2)
%     newFtimes(j)=sum([MCESorted_arry{:}]==newF(j));
%  end
%  newF=newF(newFtimes>1)';
 newF=MCESorted_arry{1}';% select lowest MCE feature combination 
 
 %%%%%%%%  FOR TESTING FEATURE UNION AND CORRELATION OUTPUT %%%%%%%%%%%%
 %%%%%%%%  cannot be used in Parfor
 test_MCE_Holdout_For_Union = crossval(classf,Std_data(:,newF),class_labels,'partition',...
        holdoutCVP)/holdoutCVP.TestSize
 corrm = corr(dataTrain(:,newF))
 

end

