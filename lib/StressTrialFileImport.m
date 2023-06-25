function [Formatted_Table,classes,ECG_col,NIBP_col,class_dpts] = StressTrialFileImport(filename,class_order,class_times)
%STRESSTRIALFILEIMPORT
% Tor Finseth
% 3/12/20
%   Configures acqknowledge exported .txt with header rows. Creates a table
%   with class labels.

% EXAMPLE INPUTS/OUTPUTS
% filename - "C:/Users/tfinseth/Documents/1 - PhD reserach/Biopac_data/P54/P54_10-31-19_trial3.txt"
% class_order - {'L';'M';'H'}
% classtimes - {1470	1834	2464	2775	2919	3227}

%% Make sure file is right type.
opts = detectImportOptions(filename);
[filepath,name,ext] = fileparts(filename);
ext_org=string(ext);
%throw error if wrong filetype
if (ext_org ~= '.csv') && (ext_org ~= '.txt')
    error ('Filetype not supported. File should be a .csv or .txt')
end
if ext_org == '.csv'
    % determine how many classes
    T = readtable(filename,opts);
    [classes,~,~] = unique(T.class,'stable');
    count = 0;
    ECG_col=1;
    disp('ECG is set as the first column.')
end
if ext_org == '.txt'
    % This is for .txt made by acqknowledge with headers
    % Check errors
        [m,n]=size(class_order);
        [m2,n2]=size(class_times);
        if (m>1 && n>1) %|| (m==1 && n==1)
                error('Wrong number of class inputs')
        end
        if (max(m2,n2)/max(m,n) ~=2)
                error('Wrong number of class times. Check that each session has a beginning and end time.')
        end
    % determine how many lines are header
    fid=fopen(filename);
    cdata=textscan(fid,'%s','delimiter',';', 'HeaderLines', 1);
    fclose(fid);
    Headvar=cdata{1,1}(3:opts.VariableNamesLine-1,1);
    Headvar=Headvar(1:2:end,:)';
    Headvar=regexprep(Headvar, '[-.,''!?/ ]', ''); %remove punctuation to create tablenames
    Headvar=regexprep(Headvar,'[0-9]',''); %remove numbers
    try
        Headvar(1,size(Headvar,2))=cellfun(@(x) x(1:3), Headvar(1,size(Headvar,2)), 'UniformOutput', false); %change name to min or sec
    catch
        error('you did not select the two boxes for "header" and "time" when saving the acqknowledge file to a .txt')
    end
    %put min/sec as first variable
    idx = 1;
    Headvar(1,idx+1:end+1) = Headvar(1,idx:end);
    Headvar(1,idx) = Headvar(1,size(Headvar,2));
    Headvar=Headvar(1,1:end-1);
    opts.VariableNames=Headvar;
    t = readtable(filename,opts);
    t=t(2:end,:);%delete 1st row
    %does the last column have numbers,otherwise delete
    TF = cellfun(@isnumeric,table2cell(t(1,:))); % look at the first row of the table, convert to a cell array temporarily, then create a logical array based on isnumeric
    if (numel(TF(TF == false)) > 0) % if any elements failed and set your flag
        %if a number of rows has zero, delete
        t(:,find(TF == 0,2)) = [];
    end
    %convert min to seconds
    if (string(t.Properties.VariableNames(1))=="min")
        t{:,1}=(t{:,1}*60);
        t.Properties.VariableNames(1)="sec";
    end
    %convert hr to seconds
    if (string(t.Properties.VariableNames(1))=="hrs")
        t{:,1}=(t{:,1}*3600);
        t.Properties.VariableNames(1)="sec";
    end
    if iscell(class_times)
        class_times=cell2mat(class_times);
    end
    %Check if the input time match the datafile
    if isempty(find(t.sec>class_times(length(class_times)-1),1))
            disp('ERROR: Are you sure you put in the right times?')
            disp('One of the classes will be missing from the final table.')
    end
    %delete rows up to the starting session
    rowend = find(t.sec>=class_times(1),1)-1;
    t((1:rowend),:) = [];
    class_var=[];
    class_var{1,1}=1;
    for N=1:1:(n2/2)
        %copy/paste the first class session
        if N==1
            class_var{1,1}=class_order(N);
        end
        rowend = find(t.sec>class_times(N*2),1);
        if isempty(find(t.sec>class_times(N*2),1)) %if the file ended before the given session time
            rowend=size(t,1);
        end
        class_dpts(N)=size((size(class_var,2)+1):rowend,2); %number of datapoints in trial
        class_var((size(class_var,2)+1):rowend)=class_order(N); %label the session
        %delete rows after session
        if N ~= (n2/2) 
            rowend2 = find(t.sec>=class_times(N*2+1),1)-1;
            t((rowend:rowend2),:) = [];
        end
    end
    %Delete extra rows in table
    t((size(class_var,2)+1:end),:) = [];
    class_var=class_var';
    
    %delete analog ECG and RSP (if we already have these as calc channels)
    count=0;
    for N=1:length(Headvar)
        if find(string(t.Properties.VariableNames{1,N}(1:3))=="ECG")==1  %find ECG column
            count=count+1;
        end
    end
    if count==2 
        try 
           t.RSPXRSPECR =[];
           t.ECGYRSPECR=[];
        catch
            %the realtime file wont have these analog channels, so it will catch everytime.
        end
    end
    if count==1
        try
        if ~isempty(t.RSPXRSPECR)
            t.RESP=t.RSPXRSPECR; %create a RSP channel to compare the SFS to the Mainprogram--- RESP is RSPcalc in biopac
            try
                t.RSPXRSPECR =[];
            catch
            end
        end
        catch
        end
    end
    
    %Create a log transformed copy of EDA, EDAtonic, pNN50, and RespirationRate
    t.EDA_Ln=log(abs(t.EDA));
    try
        t.EDAtonic_Ln=log(abs(t.EDAtonic));
   catch
        warning('EDAtonic does not exist.')
    end
    try
        t.pNN50_Ln=log(t.pNN+1);
   catch
        warning('pNN50 does not exist.')
    end
    
    %Add a class column and delete rows between sessions
    t = addvars(t,class_var,'After',size(t,2));
    t.Properties.VariableNames(end) = {'class'};
    [classes,~,~] = unique(string(t.class),'stable');
    T=t;
    
    %%Delete unneeded variables(columns), but not ECG or NIBP
    T.EDA=[]; %log transform EDA kept
    %T.EDAtonic=[];
    T.sec =[];
    T.ZXNICOR = [];
    T.dZdtXNICOR = [];
    T.EDAYPPGEDR =[];
    T.NIBP=[];
    
    %Find NIBP column for frequency analysis
    for N=1:length(T.Properties.VariableNames)
        if find(string(T.Properties.VariableNames{1,N}(1:3))=="NIB")==1  %find ECG column
            T.Properties.VariableNames(N)="NIBPD"; %rename to avoid legnth resitrctions on names in tables
            NIBP_col=N;
            break
        else 
            NIBP_col=[];
        end
    end
    %Find ECG column for HRV analysis
    for N=1:length(T.Properties.VariableNames)
        if find(string(T.Properties.VariableNames{1,N}(1:3))=="ECG")==1  %find ECG column
            ECG_col=N;
        break
        end
    end
%     T.SBP_1=[]; %This is SBP2 in Acqknowledge -- Realtime uses SBP rather than SBP2, would have to set Realtime to connect to the SBP2 channel, if you want to change this fucntion to read SBP2 from file.
%     T.DBP_1=[];
    %Final_Table.RespirationRateMean=[];
    Formatted_Table=T;
end
end

