%-------------------------------------------------------------
% print Table 1, 3 of our paper, which present the comparison between 
% RBCD, RABCD and iRBBSS
% 
% Note: if you do not generate the matfile in advance, please run ???.m
% first and set the right parameters of nRun and date_run according to the 
% settings therein. 
% 
%------------------------------
% Bo Jiang, Ya-Feng Liu, 
% A Riemannian exponential augmented Lagrangian method for computing the 
% projection robust Wasserstein distance, Accepted by NeurIPS 2023.  
%
% Author: Bo Jiang, Ya-Feng Liu, 
%   Version 0.0 ... 2022/08
%   Version 1.0 ... 2023/04 
% Contact information: jiangbo@njnu.edu.cn, yafliu@lsec.cc.ac.cn
%-------------------------------------------------------------

clear
data_type = 'shakespeare'; % Table 1
data_type = 'mnist';     % Table 3
switch data_type
    case 'shakespeare'
        scripts = {'H5', 'Ham', 'JC', 'MV', 'Oth', 'Rom'};
        nRun = 1;   % in our paper, we set nRun = 20; 
        eta = 0.1; 
        date_run = '24-Oct-2023';  
        ratio = 1;
    case 'mnist'
        scripts = {'D0','D1','D2','D3','D4','D5','D6','D7','D8','D9'};
        nRun = 1;
        date_run = '24-Oct-2023';
        eta = 8;
        ratio = 1/1000;
end
%% load the results of RBCD and RABCD
str_theta = 'theta';
path_str = 'results-2023';
str_iRBBS = '_iRBBS';
str_BCD = '_RBCD';
str_ABCD = '_RABCD';
doRABCD = 1;
doRBCD = 1;
switch data_type
    case 'mnist'
        str_ABCD = '';
        doRABCD = 0;
end

%% check whether the file exists 
nls = 0;
while 1
    matfile = sprintf('./%s/%s_nrun%d%s%s%s_eta%.3f_%s.mat',...
        path_str,data_type,nRun,str_iRBBS,str_BCD,str_ABCD,eta,date_run);
    if ~exist(matfile,'file')
        date_run = datestr(datenum(date_run) + 1);
        nls = nls + 1;
    else
        break;
    end
    if nls > 10
        break;
    end
end

%% load the results of RBCD/RABCD
if doRABCD
    load(matfile,"perf_iRBBS_all","Perf_iRBBS_cell","perf_RABCD_all","Perf_RABCD_cell",...
        "perf_RBCD_all","Perf_RBCD_cell","theta_list");
else
    load(matfile,"perf_iRBBS_all","Perf_iRBBS_cell",...
        "perf_RBCD_all","Perf_RBCD_cell","theta_list");
end
%% load the results of iRBBS
i_nd = 0;
for i = 1:length(scripts)
    art1 = scripts{i};
    for j = i+1:length(scripts)
        i_nd = i_nd + 1;
        art2 = scripts{j};
        data_name = sprintf('./%s/%s_%s.mat',data_type,art1,art2);
        load(data_name);

        X = X';
        Y = Y';
        r = r';
        c = c';
        [d,nX] = size(X);
        [~,nY] = size(Y);
        %% corresponding to iRBBSS-inf, iRBBSS-0.1, and iRBBSS-0 in our paper
        theta_list_show = [inf 0.1 0];

        nGrad_BB = zeros(length(theta_list_show),1);
        nSK_BB = zeros(length(theta_list_show),1);
        time_BB = zeros(length(theta_list_show),1);
        PRW_BB = zeros(length(theta_list_show),1);

        for i = 1:length(theta_list_show)
            temp = theta_list_show(i);
            index = find(theta_list == temp);
            temp = Perf_iRBBS_cell{i_nd}{1};
            temp = mean(temp,3);
            nGrad_BB(i) = temp(6,index);
            nSK_BB(i) = temp(7,index);
            time_BB(i)= temp(8,index);
            PRW_BB(i) = temp(11,index);
        end

        temp = mean(Perf_RBCD_cell{i_nd}{1},2);
        nGrad_BCD = temp(6);
        time_BCD = temp(7);
        PRW_BCD = temp(10);

        if doRABCD
            temp = mean(Perf_RABCD_cell{i_nd}{1},2);
            nGrad_ABCD = temp(6);
            time_ABCD = temp(7);
            PRW_ABCD = temp(10);
            perf_time = [time_BCD,time_ABCD,time_BB'];
        else
            perf_time = [time_BCD,time_BB'];
        end

        Perf_time(i_nd,:) = perf_time;
        min_time = min(perf_time);
        relative_perf_time = perf_time;%./min_time;
        %%
        if doRABCD
            str_nGrad = '&  %4.0f & %4.0f & %4.0f';
        else
            str_nGrad = '&  %4.0f & %4.0f';
        end
        for itemp = 1:length(theta_list_show) - 1
            str_nGrad = strcat(str_nGrad,'& %.0f/%.0f');
        end
        %%
        str_time = '';
        for itemp = 1:length(theta_list_show) + doRBCD + doRABCD
            if perf_time(itemp) == min_time
                if perf_time(itemp) >= 10
                    str_time = strcat(str_time,'& \\textBF{%.1f}');
                else
                    str_time = strcat(str_time,'& \\textBF{%.1f}');
                end
            else
                if perf_time(itemp) >= 10
                    str_time = strcat(str_time,'& %.1f');
                else
                    str_time = strcat(str_time,'& %.1f');
                end
            end
        end
        %%
        if doRABCD
            perf_PRW = [PRW_BCD,PRW_ABCD,PRW_BB'];
        else
            perf_PRW = [PRW_BCD,PRW_BB'];
        end
        Perf_PRW(i_nd,:) = perf_PRW;
        max_PRW = max(perf_PRW);

        relative_perf_PRW = ratio*perf_PRW; (max_PRW - perf_PRW)./max_PRW;

        str_PRW = '';
        for itemp = 1:length(theta_list_show) + doRBCD + doRABCD
            if perf_PRW(itemp) == max_PRW
                str_PRW = strcat(str_PRW,'& \\textBF{%.6f}');
            else
                str_PRW = strcat(str_PRW,'& %.6f');
            end
        end

        temp1 = nGrad_BB(2:end);
        temp2 = nSK_BB(2:end);
        temp = [temp1 temp2]; temp = temp';
        nGrad_nSK_temp = temp(:);
        if doRABCD
            perf_nGradSk = [nGrad_BCD,nGrad_ABCD,nGrad_BB(1),nGrad_nSK_temp'];
        else
            perf_nGradSk = [nGrad_BCD,nGrad_BB(1),nGrad_nSK_temp'];
        end
        Perf_nGradSk(i_nd,:) = perf_nGradSk;

        art1_s = art1;
        art2_s = art2;
        if strcmp(art1,'Rom')
            art1_s = 'RJ';
        end
        if strcmp(art1,'Ham')
            art1_s = 'H';
        end
        if strcmp(art1,'Oth')
            art1_s = 'O';
        end

        if strcmp(art2,'Rom')
            art2_s = 'RJ';
        end
        if strcmp(art2,'Ham')
            art2_s = 'H';
        end
        if strcmp(art2,'Oth')
            art2_s = 'O';
        end

        fprintf(1, strcat('%s/%s',str_PRW,str_nGrad,str_time,' \\\\\n'),...
            art1_s,art2_s,relative_perf_PRW,...
            perf_nGradSk,...
            relative_perf_time);
        %         'Perf_iRBBS_cell','Perf_BCD_cell','Perf_ABCD_cell','eta',...
        %             'max_subiter_sk_List','scripts'
    end
end



str_PRW = '';

temp = max(mean(Perf_PRW));
mean_perf_PRW = ratio*mean(Perf_PRW); (temp - mean(Perf_PRW))./temp;


for itemp = 1:length(theta_list_show) + doRABCD + doRBCD
    if mean_perf_PRW(itemp) == max(mean_perf_PRW)
        str_PRW = strcat(str_PRW,'& \\textBF{%.6f}');
    else
        str_PRW = strcat(str_PRW,'& %.6f');
    end
end

str_time = '';
temp = min(mean(Perf_time));
mean_perf_time = mean(Perf_time);
for itemp = 1:length(theta_list_show) + doRBCD + doRABCD
    if mean_perf_time(itemp) == min(mean_perf_time)
        if min(mean_perf_time) >= 1
            str_time = strcat(str_time,'& \\textBF{%.1f}');
        else
            str_time = strcat(str_time,'& \\textBF{%.2f}');
        end
    else
        if mean_perf_time(itemp)  >= 1
            str_time = strcat(str_time,'& %.2f');
        else
            str_time = strcat(str_time,'& %.2f');
        end
    end
end
if doRABCD
    fprintf(1,' \\cmidrule(l){2-16} \n');
else
    fprintf(1,' \\cmidrule(l){2-13} \n');
end

if doRABCD
    str_nGrad = '&  %4.0f & %4.0f & %4.0f';
else
    str_nGrad = '&  %4.0f & %4.0f';
end

for itemp = 1:length(theta_list_show) - 1
    str_nGrad = strcat(str_nGrad,'& %.0f/%.0f');
end

fprintf(1, strcat('AVG',str_PRW,str_nGrad,str_time,'\\\\ \n'),...
    mean_perf_PRW,mean(Perf_nGradSk),mean(Perf_time)); %./min(mean(Perf_time))

