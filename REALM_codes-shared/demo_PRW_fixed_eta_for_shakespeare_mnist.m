%-------------------------------------------------------------
% test R(A)BCD, iRBBSS for shakespeare and mnist datasets in our paper
%
% Suppose data_type = 'shakespeare'; and set nRun = 1; after running this codes,
% you will obtain a matfile, named,
% 'shakespeare_nrun1_iRBBS_RBCD_RABCD_eta0.100_24-Oct-2023.mat',
% in the folder ``results-2023'';
% To generate Tables 1 and 3 in our paper, run
% 'Tables1_3_shakespeare_mnist_iRBBS_RABCD.m' with the following settings
%     nRun = 1;
%     date_run = '24-Oct-2023';
%     data_type = 'shakespeare';
%------------------------------
% Bo Jiang, Ya-Feng Liu,
% A Riemannian exponential augmented Lagrangian method for computing the
% projection robust Wasserstein distance, Accepted by NeurIPS 2023.
%
% Author: Bo Jiang, Ya-Feng Liu,
%   Version 0.0 ... 2022/08
%   Version 1.0 ... 2023/04
%-------------------------------------------------------------
clear
T00 = tic;
nRun = 1; 20; % in our paper, we set nRun  = 20;
data_type = 'shakespeare';
data_type = 'mnist';
switch data_type
    case 'mnist'
        scripts = {'D0','D1','D2','D3','D4','D5','D6','D7','D8','D9'};
        eta =  8;  % can also try other eta_List, e.g., [8   7   6   5   4    3];
    case  'shakespeare'
        scripts = {'H5', 'Ham', 'JC', 'MV', 'Oth', 'Rom'};
        eta  =  0.1;
end
%% Run the test procedure
Run_test(nRun,data_type,scripts,eta);
TSOLVE = toc(T00)

%% function
function  Run_test(nRun,data_type,scripts,eta)
k = 2;
dfile=sprintf('log/%s_%s_nRun%d.log', data_type,datestr(now,'yyyy-mm-dd-HH'),nRun);
diary(dfile);
diary on;
T0 = tic;
opts.data_type = data_type;
rng(202204232);
seedList = randi(2^32,max(nRun,100),1);
path_str = 'results-2023';

fprintf(1,'===========================  eta = %.3f ======================\n',...
    eta);


theta_list = [0 0.1 0.2 0.4 inf];
%% settings for whether running the methods
doiRBBS  = 1;
if strcmp(data_type,'mnist')
    doRBCD = 1;
    doRABCD = 0;
else
    doRBCD  = 1;
    doRABCD = 1;
end
doPost  = 1;

str_iRBBS = ''; str_BCD  = ''; str_ABCD = '';

if doiRBBS; str_iRBBS = '_iRBBS'; end
if doRBCD; str_BCD = '_RBCD';  end
if doRABCD; str_ABCD = '_RABCD'; end

matfile = sprintf('./%s/%s_nrun%d%s%s%s_eta%.3f_%s.mat',...
    path_str,data_type,nRun,str_iRBBS,str_BCD,str_ABCD,eta,date);
%% settings of opts
opts.doiRBBS = doiRBBS;
opts.doRABCD= doRABCD;
opts.doRBCD = doRBCD;
opts.doPost = doPost;
opts.theta_list  = theta_list;

i_nd = 0; % the number of instances
n_scripts = length(scripts);
total_nd = n_scripts*(n_scripts-1)/2;
Perf_iRBBS_cell = cell(total_nd,1);
Perf_RBCD_cell = cell(total_nd,1);
Perf_RABCD_cell = cell(total_nd,1);

for i = 1:n_scripts
    art1 = scripts{i};
    for j = i+1:n_scripts
        i_nd = i_nd + 1;
        art2 = scripts{j};
        data_name = sprintf('./%s/%s_%s.mat',data_type,art1,art2);
        S = load(data_name);
        X = double(S.X');
        Y = double(S.Y');
        r = double(S.r');
        c = double(S.c');
        [d,nX] = size(X);
        [~,nY] = size(Y);
        opts.art1 = art1;
        opts.art2 = art2;
        fprintf(1,strcat('\n\n====== start of PRW fixed eta:%s\\_%s,',...
            'nX = %d, nY = %d, d = %d, k = %d ======\n\n'),...
            art1,art2,nX,nY,d,k);
        %% stepsize for RBCD and RABCD
        if opts.doRBCD || opts.doRABCD
            switch data_type
                case 'shakespeare'
                    tau_RBCD = 0.1;
                    tau_RABCD = 0.0025;
                    if strcmp(art2,'Rom')
                        tau_RABCD = 0.0015;
                        if strcmp(art1,'Ham')
                            tau_RBCD = 0.09;
                            tau_RABCD = 0.001;
                        end
                    end
                case 'mnist'
                    tau_RBCD  = 0.004;
                    tau_RABCD = 0.004;
            end
            opts.tau_RBCD = tau_RBCD;
            opts.tau_RABCD = tau_RABCD;
        end
        %% Compare different solvers
        n_theta = length(theta_list);
        Perf_iRBBS = zeros(14,n_theta,nRun);
        Perf_RBCD = zeros(13,nRun);
        Perf_RABCD = zeros(13,nRun);
        % parameters for printing
        str_num = ['%6s','&%10.3f','&%8.1e','&%8.1e','&%10.0f','&%10.0f','&%10.0f',...
            '&%10.2f','&%10.2f', '& %16.14f','& %16.14f','& %16.14f','& %16.14f','\\\\ \n'];
        stra_latex = [ '%6s','%15s','%10s','%12s','%12s','%12s',...
            '%10s','%15s','%15s', '%15s','%15s','%15s','%15s', '%16s \n'];
        str_head = sprintf(stra_latex,'alg&', 'eta&',  'gtolU&','gtol\_ab&', 'iter&','nfe&', 'sk&', ...
            'time&', 'time\_post&', 'PRW&', 'PRW\_post&','obj&','obj\_post', '\\ ');
        iRBBS_str = 'iRBBSS';
        for iRun = 1:nRun
            %% after choosing the parameters, run the comparison code
            [perf_iRBBS,perf_RBCD,perf_RABCD] = ...
                Compare_iRBBS_RABCD(X,Y,r,c,k,eta,seedList(iRun),opts);
            %% save results
            if doiRBBS
                Perf_iRBBS(:,:,iRun) = perf_iRBBS;
            end
            if doRBCD
                Perf_RBCD(:,iRun)  = perf_RBCD;
            end
            if doRABCD
                Perf_RABCD(:,iRun) = perf_RABCD;
            end
            %% print information of each iteration
            fprintf('iRun: %d/%d, id: %d\n',iRun,nRun,get(getCurrentTask(),'ID'));
            fprintf(1,'%s',str_head);
            if opts.doiRBBS
                for i_theta = 1:n_theta
                    method_str = sprintf('%s-%.1e',iRBBS_str,perf_iRBBS(1,i_theta));
                    fprintf(1,str_num,method_str,perf_iRBBS(2:13,i_theta));
                end
            end
            if opts.doRBCD
                fprintf(1,str_num,'RBCD',perf_RBCD(1:12));
            end
            if opts.doRABCD
                fprintf(1,str_num,'RABCD',perf_RABCD(1:12));
            end
        end

        Perf_iRBBS_cell{i_nd}  = {Perf_iRBBS,art1,art2,eta};
        Perf_RBCD_cell{i_nd}  = {Perf_RBCD,art1,art2,eta};
        Perf_RABCD_cell{i_nd} = {Perf_RABCD,art1,art2,eta};

        %% summary of results
        fprintf('\n summary: %s\\_%s,nX = %d, nY = %d, d = %d, k = %d \n',...
            art1,art2,nX,nY,d,k);
        fprintf('%s',str_head);
        if opts.doiRBBS
            temp = mean(Perf_iRBBS,3);
            for i_theta = 1:n_theta
                method_str = sprintf('%s-%.1e',iRBBS_str,temp(1,i_theta));
                fprintf(1,str_num,method_str,temp(2:13,i_theta));
            end
        end
        if opts.doRBCD
            temp = mean(Perf_RBCD,2);
            fprintf(1,str_num,'RBCD',temp(1:12));
        end
        if opts.doRABCD
            temp = mean(Perf_RABCD,2);
            fprintf(1,str_num,'RABCD',temp(1:12));
        end
    end
end

fprintf('\n \n summary of  %s, nRun: %d\n',data_type,nRun);
perf_iRBBS_all = zeros(14,n_theta);
%% save the results to matfiles
if doiRBBS
    for i_temp = 1:total_nd
        temp = Perf_iRBBS_cell{i_temp}{1}; %
        perf_iRBBS_all = perf_iRBBS_all + mean(temp,3);
    end
    perf_iRBBS_all = perf_iRBBS_all./total_nd;
    for i_theta = 1:n_theta
        method_str = sprintf('%s-%.1e',iRBBS_str,perf_iRBBS_all(1,i_theta));
        fprintf(1,str_num,method_str,perf_iRBBS_all(2:13,i_theta));
    end
    save(matfile,'Perf_iRBBS_cell','perf_iRBBS_all','theta_list');
end
if doRBCD
    perf_RBCD_all = zeros(13,1);
    for i_temp = 1:total_nd
        temp = Perf_RBCD_cell{i_temp}{1}; %
        perf_RBCD_all = perf_RBCD_all + mean(temp,2);
    end
    perf_RBCD_all = perf_RBCD_all./total_nd;
    fprintf(1,str_num,'RBCD',perf_RBCD_all(1:12));
    save(matfile, 'Perf_RBCD_cell','perf_RBCD_all','-append');
end
if doRABCD
    perf_RABCD_all = zeros(13,1);
    for i_temp = 1:total_nd
        temp = Perf_RABCD_cell{i_temp}{1}; %
        perf_RABCD_all = perf_RABCD_all + mean(temp,2);
    end
    perf_RABCD_all = perf_RABCD_all./total_nd;
    fprintf(1,str_num,'RABCD',perf_RABCD_all(1:12));
    save(matfile,'Perf_RABCD_cell','perf_RABCD_all','-append');
end
T_solve = toc(T0);
fprintf(1,'total running time is %4.2f mins\n',T_solve/60);
diary off;
end