%-------------------------------------------------------------
% test  different versions of ReALM for shakespeare and mnist datasets in our paper
%
% Suppose data_type = 'shakespeare'; and set nRun = 1; after running this codes,
% you will obtain a matfile, named,
% 'shakespeare_nrun1_iRBBS_RBCD_RABCD_eta0.100_24-Oct-2023.mat'  ????????? 
% in the folder ``results-2023'';
% 
% To generate Tables 2 and 5 in our paper,  run
% 'Tables2_5_shakespeare_mnist_REALM.m' with the following settings
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
%   Version 1.0 ... 2023/10
%-------------------------------------------------------------

clear
data_type = 'shakespeare';
data_type = 'mnist';
switch data_type
    case 'shakespeare'
        scripts = {'H5', 'Ham', 'JC', 'MV', 'Oth', 'Rom'};
        scripts = {'H5', 'Ham', 'JC'};
        %%  eta0, eta_min, ratio_U, ratio_ab, ratio_eta, ratio_CPL, max_update_Pi
        param_list = ...
            [20  0.007 0.25 0.25 0.25 0.9, 7;
            20  0.0035 0.25 0.25 0.25 0, 0]; 
        % if ratio_CPL == 0, then it means that ReALM reduces to the
        % Riemannian exponential penalty method, wherein the
        % penalty/regularization parameter, namely, eta,  are updated dynamically. This
        % is always better than solving a penalty problem with small eta. 
    case 'mnist'
        scripts = {'D0','D1','D2','D3','D4','D5','D6','D7','D8','D9'};
        scripts = {'D0','D1','D2'};
        param_list = [200  3 0.25 0.25 0.25 0.9 7;
            200  1 0.25 0.25 0.25 0 0]; 
end
n_param = size(param_list,1);
T0 = tic;
%% seed and nRun information
nRun = 1; 20;
rng(202204232);
seedList = randi(2^32,nRun,1);
record = 1;
%% reduced dimension
k = 2;
stra = ['%6s','%10s','%12s','%12s','%10s','%10s','%15s', '%12s', '%22s',...
     '%22s','%22s', '%22s', '%12s', '%12s',   '\n'];
str_num = strcat('%6s   %4.3e    %2.1e    %2.1e     %5.0f     %5.0f   %5.0f',...
    '%5.0f     %2.1e  %8.6e  %8.6e   %8.6e   %8.6e  %10d  %10.1f  %4.2f\n');
%% log files
dfile = sprintf('log/ReALM_iRBBS_%s_%s_nRun%d.log',data_type,datestr(now,'yyyy-mm-dd'),nRun);
diary(dfile);
diary on;
filename = sprintf('./results-2023/%s_nrun%d_LatexResults_ReALM_%s_v2.txt',data_type,nRun,date);
matfile  = sprintf('./results-2023/%s_nrun%d_Results_ReALM_iRBBS_%s.mat',data_type,nRun,date);
fid_latex = fopen(filename,'w+');
%% number of problems
n_ind = (length(scripts) - 1)*length(scripts)/2;
i_nd = 0;
%% performance
Perf_ReALM_cell = cell(n_ind,n_param);
save(matfile,'param_list');
%% start to solve each instance
for i = 1:length(scripts)
    art1 = scripts{i};
    for j = i+1:length(scripts)
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
        stra_latex = ['%6s','%10s','%12s','%12s','%10s','%10s','%15s', '%12s', '%22s', '%22s' '\n'];
        str_head = sprintf(stra_latex,'alg&', 'eta&', 'nfe&', 'sk&', ...
            'time&',   'PRW&', 'PRW\_post&', 'outiter&', '\#Pi &', 'post_time\\ \hline');
        fprintf(fid_latex,'%s',str_head);
        fprintf(1,'\n\n====== start of ReALM: nX = %d, nY = %d, d = %d, k = %d ======\n\n',nX,nY,d,k);
        fprintf(1,'%s_%s\n',art1,art2);
        fprintf(fid_latex,'\\multicolumn{10}{|c|}{nX = %d, nY = %d, d = %d}\\\\ \\hline \n',nX,nY,d);
        Perf_ReALM = zeros(nRun,15,n_param);
        nan_ReALM = 0;
        for iRun = 1:nRun
            fprintf(1,'%d/%d\n',iRun,nRun);
            str_head = sprintf(stra,'alg', 'eta', 'nrm_gU', 'tol_ab', 'iter', 'nfe', 'sk_iter', ...
                'time', 'PRW', 'obj', 'obj_post', 'outiter', 'update_Pi',  'post_time');
            fprintf(1,'%s',str_head);
            %% generate initial points
            seed = seedList(iRun);
            a0 = zeros(nX,1);
            b0 = zeros(nY,1);
            U0 = generate_initial_U(seed,X,Y,r,c,nX,nY,k);
            Pi0 = ones(nX,nY);
            Xt = X';
            C0 = (-2*X')*Y + sum(Xt.^2,2) + sum(Y.^2,1);
            max_XY = max(C0(:)); % for scaled Riemannian gradient tolerance
            gtol_ab = 1e-6*max(norm(r,'inf'),norm(c,'inf'));
            gtol_U =  2*max_XY*gtol_ab;
            ones_nY = ones(nY,1);
            ones_nX = ones(nX,1);
            %% ReALM
            for i_param = 1:n_param
                eta0    = param_list(i_param,1);
                eta_min = param_list(i_param,2);
                ratio_U = param_list(i_param,3);
                ratio_ab = param_list(i_param,4);
                ratio_eta = param_list(i_param,5);
                ratio_CPL = param_list(i_param,6);
                max_update_Pi = param_list(i_param,7);
                opts.gtol_ab_sub = 1e-1*max(norm(r,'inf'),norm(c,'inf')); 1e-3;
                opts.gtol_U_sub  = 2*max_XY*opts.gtol_ab_sub;
                %% stopping tolerance for our iterations
                opts.tol_CPL  = 1e-3; % tolerance for CPLmentarity
                opts.gtol_U   = gtol_U;
                opts.gtol_ab  = gtol_ab;
                %% decrease factors for U, ab and eta
                opts.ratio_U   = ratio_U; 
                opts.ratio_ab  = ratio_ab;  
                opts.ratio_eta = ratio_eta; 
                opts.ratio_CPL = ratio_CPL; 
                opts.theta_exp = 0.1;
                opts.theta_log = 10;
                opts.maxiter =  30;
                opts.record = record;
                opts.max_update_Pi = max_update_Pi;
                %% parameters for inner solvers
                opts.maxSubiter = 1000;
                opts.record_sub = 0;
                opts.theta = 1;
                opts.eta_min = eta_min;
                %% call the main subroutine
                t0 = tic;
                [Pi_ReALM,U_ReALM,a_ReALM,b_ReALM,out_ReALM] = ReALM4PRW(X,Y,r,c,Pi0,a0,b0,U0,eta0,opts);
                t_ReALM = toc(t0);
                PRW_ReALM = out_ReALM.PRW;
                f_ReALM   = out_ReALM.f;
                eta_ReALM = out_ReALM.eta;
                CPL_ReALM = out_ReALM.CPL;
                nrm_gradU_ReALM = out_ReALM.nrm_gradU;
                feasi_ab_ReALM = out_ReALM.feasi_ab;

                outiter_ReALM = out_ReALM.iter;
                iter_sk_ReALM = out_ReALM.total_iter_sk;
                subiter_ReALM = out_ReALM.total_subiter;

                n_exp_ReALM   = out_ReALM.n_exp; %
                n_log_ReALM   = out_ReALM.n_log; %

                nfe_ReALM = out_ReALM.total_nfe;
                num_update_Pi_ReALM = out_ReALM.num_update_Pi;

                %% postprocessing
                if ~sum(isnan(U_ReALM))
                    t0 = tic;
                    if strcmp(art1,'TM') && strcmp(art2,'T') || strcmp(art1,'I') && strcmp(art2,'T')
                        [Pi_ReALM_post,a_ReALM_post,b_ReALM_post] = PRW_fixedU_Mosek(U_ReALM,X,Y,r,c);
                    else
                        [Pi_ReALM_post,a_ReALM_post,b_ReALM_post] = PRW_fixedU_Gurobi(U_ReALM,X,Y,r,c,0);
                    end
                    t_ReALM_post = toc(t0);
                    C_ReALM = compute_C(U_ReALM,X,Y);
                    feasi = norm(sum(Pi_ReALM_post,2) - r,1) + norm(sum(Pi_ReALM_post,1) - c',1);
                    PRW_ReALM_post = sum(sum(C_ReALM.*Pi_ReALM_post));
                    f_ReALM_post = r'*a_ReALM_post + c'*b_ReALM_post + min(min(C_ReALM + a_ReALM_post + b_ReALM_post'));
                else
                    f_ReALM_post = f_ReALM;
                    nan_ReALM = nan_ReALM + 1;
                end
                Perf_ReALM(iRun,:,i_param) = [eta_ReALM,nrm_gradU_ReALM,feasi_ab_ReALM, subiter_ReALM,...
                    nfe_ReALM,n_exp_ReALM,n_log_ReALM, t_ReALM, PRW_ReALM,PRW_ReALM_post,f_ReALM, f_ReALM_post,...
                    outiter_ReALM,num_update_Pi_ReALM,t_ReALM_post];
            end
            %% print the information of iRun
            for i_param = 1:n_param
                method_name = sprintf('ALM%d',i_param);
                fprintf(1,str_num,method_name,Perf_ReALM(iRun,:,i_param));
            end
        end

        %% print the averaged information over nRun for an instance 
        fprintf(1,'\nsummary: %s_%s\n',art1,art2);
        stra_temp = ['%6s','%10s','%12s','%12s','%10s','%10s','%10s','%15s', '%12s', '%22s',...
            '%22s','%22s', '%22s', '%12s', '%12s', '%12s'];
        str_head = sprintf(stra_temp,'alg', 'eta', 'nrm_gU', 'tol_ab', 'iter', 'nfe', 'n_exp', 'n_log', ...
            'time', 'PRW', 'PRW_post', 'obj', 'PRW_post', 'outiter', 'update_Pi',  'post_time');
        fprintf(1,'%s\n',str_head);
        for i_param = 1:n_param
            method_name = sprintf('ALM%d',i_param);
            fprintf(1,str_num,method_name,mean(Perf_ReALM(:,:,i_param),1));
        end

        str_num_latex = '%6s & %4.3f &  %5.0f &  %5.0f  & %2.1e  &  %8.6e &  %8.6e   & %2.1f & %2.1f & %4.2f \\\\ \n';
        for i_param = 1:n_param
            method_name = sprintf('ALM%d',i_param);
            temp = mean(Perf_ReALM(:,:,i_param),1);
            fprintf(fid_latex,str_num_latex,'ReALM',temp([1 5 6 7 8 9 13 14 15]));
        end

        fprintf(fid_latex,'\\hline \n');
        for i_param = 1:n_param
            Perf_ReALM_cell{i_nd,i_param} = Perf_ReALM(:,:,i_param);
        end
        %% save matfile 
        save(matfile,'scripts','Perf_ReALM_cell','-append');
    end
end
Tsolve = toc(T0)
fprintf(1,'total running time is %4.2f mins\n',Tsolve/60);
fprintf(1,'\n\n average performance \n\n');

%% print the summary information of all instances
for i_param = 1:n_param
    perf_ReALM_temp = zeros(n_ind,15);
    for i = 1:n_ind
        perf_ReALM_temp(i,:) = mean(Perf_ReALM_cell{i,i_param},1);
    end
    method_name = sprintf('ALM%d',i_param);
    fprintf(1,str_num,method_name,mean(perf_ReALM_temp,1));
end

