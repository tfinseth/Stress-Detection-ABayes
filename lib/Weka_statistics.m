function [accuracy_out,metrics] = Weka_statistics(CM,classNames)

alpha=0.05;

p = inputParser;
addRequired(p,'CM',@(CM) validateattributes(CM,{'numeric'},{'square','nonempty','integer','real','finite','nonnan','nonnegative'}));
parse(p,CM);
x=p.Results.CM;
clear p default* validation*

m=size(x,1);
tr=repmat('-',1,80);

f=diag(ones(1,m)); %unweighted

disp('=== Summary ===')
disp(' ')
%disp(tr)
kcomp;
disp(' ')

function kcomp
        n=sum(x(:)); %Sum of Matrix elements
        x=x./n; %proportion
        r=sum(x,2); %rows sum
        s=sum(x); %columns sum
        Ex=r*s; %expected proportion for random agree
        pom=sum(min([r';s])); %maximum proportion observable
        po=sum(sum(x.*f)); %proportion observed
        pe=sum(sum(Ex.*f)); %proportion expected
        k=(po-pe)/(1-pe); %Cohen's kappa
        km=(pom-pe)/(1-pe); %maximum possible kappa, given the observed marginal frequencies
        ratio=k/km; %observed as proportion of maximum possible
        sek=sqrt((po*(1-po))/(n*(1-pe)^2)); %kappa standard error for confidence interval
        ci=k+([-1 1].*(abs(-realsqrt(2)*erfcinv(alpha))*sek)); %k confidence interval
        wbari=r'*f;
        wbarj=s*f;
        wbar=repmat(wbari',1,m)+repmat(wbarj,m,1);
        a=Ex.*((f-wbar).^2);
        var=(sum(a(:))-pe^2)/(n*((1-pe)^2)); %variance
        z=k/sqrt(var); %normalized kappa
        p=(1-0.5*erfc(-abs(z)/realsqrt(2)))*2;
        
        c_matrix=CM;
        [row,col]=size(c_matrix);
            if row~=col
                error('Confusion matrix dimention is wrong')
            end
            n_class=row;
            switch n_class
                case 2
                    TP=c_matrix(1,1);
                    FN=c_matrix(1,2);
                    FP=c_matrix(2,1);
                    TN=c_matrix(2,2);
                    
                otherwise
                    TP=zeros(1,n_class);
                    FN=zeros(1,n_class);
                    FP=zeros(1,n_class);
                    TN=zeros(1,n_class);
                    for i=1:n_class
                        TP(i)=c_matrix(i,i);
                        FN(i)=sum(c_matrix(i,:))-c_matrix(i,i);
                        FP(i)=sum(c_matrix(:,i))-c_matrix(i,i);
                        TN(i)=sum(c_matrix(:))-TP(i)-FP(i)-FN(i);
                    end
                    
            end
            
            
            %Calulations
            %1.P-Positive
            %2.N-Negative
            %3.acuuracy
            %4.error
            %5.Sensitivity (Recall or True positive rate)
            %6.Specificity
            %7.Precision
            %8.FPR-False positive rate
            %9.F_score
            %10.MCC-Matthews correlation coefficient
            %11.kappa-Cohen's kappa
            P=TP+FN;
            N=FP+TN;
            switch n_class
                case 2
                    accuracy=(TP+TN)/(P+N);
                    Error=1-accuracy;
                    Result.Accuracy=(accuracy);
                    Result.Error=(Error);
                otherwise
                    accuracy=(TP)./(P+N);
                    Error=(FP)./(P+N);
                    Result.Accuracy=sum(accuracy);
                    Result.Error=sum(Error);
            end
            RefereceResult.AccuracyOfSingle=(TP ./ P)';
            RefereceResult.ErrorOfSingle=1-RefereceResult.AccuracyOfSingle;
            Sensitivity=TP./P; %Recall
            Specificity=TN./N;
            Precision=TP./(TP+FP); %PPV
            TPR=Specificity;
            FPR=1-Specificity;
            beta=1;
            F1_score=( (1+(beta^2))*(Sensitivity.*Precision) ) ./ ( (beta^2)*(Precision+Sensitivity) );
            MCC=[( TP.*TN - FP.*FN ) ./ ( ( (TP+FP).*P.*N.*(TN+FN) ).^(0.5) );...
                ( FP.*FN - TP.*TN ) ./ ( ( (TP+FP).*P.*N.*(TN+FN) ).^(0.5) )] ;
            MCC=max(MCC);
            %https://lettier.github.io/posts/2016-08-05-matthews-correlation-coefficient.html
            %if there is only 1 class in the testing dataset, you will get MCC=nan
            
            %Kappa Calculation BY 2x2 Matrix Shape
            pox=sum(accuracy);
            Px=sum(P);TPx=sum(TP);FPx=sum(FP);TNx=sum(TN);FNx=sum(FN);Nx=sum(N);
            pex=( (Px.*(TPx+FPx))+(Nx.*(FNx+TNx)) ) ./ ( (TPx+TNx+FPx+FNx).^2 );
            kappa_overall=([( pox-pex ) ./ ( 1-pex );( pex-pox ) ./ ( 1-pox )]);
            kappa_overall=max(kappa_overall);
            
            %Kappa Calculation BY n_class x n_class Matrix Shape
            po=accuracy;
            pe=( (P.*(TP+FP))+(N.*(FN+TN)) ) ./ ( (TP+TN+FP+FN).^2 );
            kappa=([( po-pe ) ./ ( 1-pe );( pe-po ) ./ ( 1-po )]);
            kappa=max(kappa);
        
        for i = 1:m
            %FNR(i) = CM(1,2)/sum(CM(:,2));
            %FPR(i) = CM(2,1)/sum(CM(:,1));
            TPR(i) = CM(i,i)/sum(CM(i,:)); %,Sens
            %TNR(i) = CM(2,2)/sum(CM(:,2));
            PPV(i) = CM(i,i)/sum(CM(:,i)); % Precision
            
            Sens(i) = CM(i,i)/sum(CM(i,:));
            Fmeasure(i) = 2/((1/Sens(i))+(1/PPV(i)));
            %Spec(i) = CM(2,2)/(sum(CM(i+1,:)));
        end
        
        
        %display results
        fprintf('Correctly Classified Instances (Accuracy)        %1.0f       %0.4f %%\n',Result.Accuracy*n,Result.Accuracy*100);
        fprintf('Incorrectly Classified Instances (Error)         %1.0f        %0.4f %%\n',Result.Error*n,Result.Error*100);
        %fprintf('Kappa statistic                                  %0.2f\n',kappa_overall)
        %fprintf('Mean absolute error (kappa error)                %0.4f\n',sek)
        fprintf('Total Number of Instances                        %1.0f\n', n)
        
        %disp(' ')
        %fprintf('Random agreement (pe) = %0.4f\n',pe)
        
        
        %fprintf('Random agreement (pe) = %0.4f\n',pe)
        %fprintf('Agreement due to true concordance (po-pe) = %0.4f\n',po-pe)
        %fprintf('Residual not random agreement (1-pe) = %0.4f\n',1-pe)
        %fprintf('kappa C.I. (alpha = %0.4f) = %0.4f     %0.4f\n',alpha,ci)
        %fprintf('Maximum possible kappa, given the observed marginal frequencies = %0.4f\n',km)
        %fprintf('k observed as proportion of maximum possible = %0.4f\n',ratio)
        disp(' ')
        disp('=== Detailed Accuracy By Class ===')
        disp(' ')
        
        %fa = randi(4,13,9);                     % Create Data
        %fprintf(1, '      TP Rate    Precision    Recall    F-Measure    Class\n')    % Column Titles
        %fprintf(1, '\t\t%0.2f \t\t%0.2f \t\t%0.2f \t\t%0.2f \t\t%s\n', [TPR;PPV;Sens;Fmeasure;classNames])      % Write Rows
        classname = cellstr(classNames);
       
        if size(TPR,2)==length(classname)
            T=table(TPR',PPV',Sens',Fmeasure','RowNames', classname); %2/12/19 - change classNames to classname because of error that said that 'RowNames' needed to be a cell array.
            T.Properties.VariableNames = {'TP_Rate' 'Precision' 'Recall' 'F_Measure'};
            disp(T)
        end
        disp(' ')
        disp('=== Detailed Accuracy, Weighted Mean ===')
        disp(' ')
        fprintf('Accuracy                   %0.4f\n',Result.Accuracy)
        fprintf('Error                      %0.4f\n',Result.Error)
        fprintf('TP Rate                    %0.4f\n',nanmean(TPR))
        fprintf('FP Rate                    %0.4f\n',nanmean(FPR))
        fprintf('Precision                  %0.4f\n',nanmean(Precision))
        fprintf('Recall (sensitivity)       %0.4f\n',nanmean(Sensitivity))
        fprintf('Specificity                %0.4f\n',nanmean(Specificity))
        fprintf('F-Measure                  %0.4f\n',nanmean(F1_score))
        fprintf('MCC                        %0.4f\n',nanmean(MCC))
        fprintf('Kappa statistic            %0.4f\n',kappa_overall)
        %fprintf('ROC Area                   %0.2f\n',k)
        %fprintf('RPC Area                   %0.2f\n',k)
        accuracy_out=Result.Accuracy;
        metrics=[Result.Accuracy,Result.Error,nanmean(TPR),nanmean(FPR),nanmean(Precision),nanmean(Sensitivity),nanmean(Specificity),nanmean(F1_score),nanmean(MCC),kappa_overall];
        disp(' ')
        disp(' ')
        
      
        disp(' ')
        disp('=== Confusion Matrix ===')
        disp(' ')
        disp(' col = predicted, row= actual ')
        CM = num2cell(CM);
        if size(CM,2)==length(classNames)
            try
                CMN = [CM classNames]; 
                disp(CMN)
            catch
                CMN=CM;
                disp(CMN)
            end
        end
        
        %{
        if k<0
            disp('Poor agreement')
        elseif k>=0 && k<=0.2
            disp('Slight agreement')
        elseif k>=0.21 && k<=0.4
            disp('Fair agreement')
        elseif k>=0.41 && k<=0.6
            disp('Moderate agreement')
        elseif k>=0.61 && k<=0.8
            disp('Substantial agreement')
        elseif k>=0.81 && k<=1
            disp('Perfect agreement')
        end
        fprintf('Variance = %0.4f     z (k/sqrt(var)) = %0.4f    p = %0.4f\n',var,z,p)
        if p<alpha
            disp('Reject null hypotesis: observed agreement is not accidental')
        else
            disp('Accept null hypotesis: observed agreement is accidental')
        end
        %}
    end
end