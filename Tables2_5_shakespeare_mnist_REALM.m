%-------------------------------------------------------------
% print Table 2 and Table 5 of our paper, which present the comparison between 
% REALM
% 
%
%------------------------------
% Reference: 
% Bo Jiang, Ya-Feng Liu, 
% A Riemannian exponential augmented Lagrangian method for computing the 
% projection robust Wasserstein distance, Accepted by NeurIPS 2023. 
%
% Author: Bo Jiang, Ya-Feng Liu, 
%   Version 0.0 ... 2022/08
%   Version 1.0 ... 2023/10
% Contact information: jiangbo@njnu.edu.cn, yafliu@lsec.cc.ac.cn
%-------------------------------------------------------------
data_type = 'shakespeare'; % Table 1
% data_type = 'mnist';       % Table 3
diff = 0.005; 
diff = 0; 
% If we set diff = 0.005, we only present instances (for space consideration) 
% where one method can return the value larger than 1.005 times of the smaller 
% one of the two values. To show the full results, please set diff = 0; 

switch data_type
    case 'mnist'
        scripts = {'D0','D1','D2','D3','D4','D5','D6','D7','D8','D9'};
        scripts = {'D0','D1','D2'};
        scale = 1/1000;  % this scale is for ease of presentation
        date =  '24-Oct-2023';
        nRun = 1; 20; 
        index_1 = 1;
        index_2 = 2;
    case 'shakespeare'
        scripts = {'H5', 'Ham', 'JC', 'MV', 'Oth', 'Rom'};
        scripts = {'H5', 'Ham', 'JC'};
        scale = 1;
        date =  '24-Oct-2023';
        nRun = 1; 20; 
        index_1 = 1;
        index_2 = 2;
end

matfile = sprintf('./results-2023/%s_nrun%d_Results_ReALM_iRBBS_%s.mat',data_type,nRun,date);
load(matfile); 

i_nd = 0;
fid = 1;
Perf_ALM = zeros(1,15);
Perf_iRBBS = zeros(1,15); 
for i = 1:length(scripts)
    art1 = scripts{i};
    for j = i+1:length(scripts)
        i_nd = i_nd + 1;
        art2 = scripts{j};
        perf_ALM = mean(Perf_ReALM_cell{i_nd,index_1},1);
        perf_iRBBS = mean(Perf_ReALM_cell{i_nd,index_2},1);  % check the parameters 

        Perf_ALM = Perf_ALM + perf_ALM; 
        Perf_iRBBS = Perf_iRBBS + perf_iRBBS; 
        str = '%s/%s &'; 
        base = (1 + diff)*min(perf_iRBBS(10),perf_ALM(10));
        if perf_iRBBS(10) > base
            str_PRW = '\\textBF{%.5f} & %.5f &';
            latex_show = 1; 
        elseif base < perf_ALM(10)
            str_PRW = '%.5f & \\textBF{%.5f} &';
            latex_show = 1; 
        else
            str_PRW = '%.5f & %.5f &';
            latex_show = 0; 
        end
        PRW_latex = [perf_iRBBS(10) perf_ALM(10)].*scale; 
        str_nGrad = '%.0f & %.0f &'; 
        nGrad_latex = [perf_iRBBS(5) perf_ALM(5)];
        str_SK = '%.0f/%.0f & %.0f/%.0f &'; 
        SK_exp_log_latex = [perf_iRBBS(6) perf_iRBBS(7) perf_ALM(6) perf_ALM(7)]; 
        str_time = '%.1f & %.1f &'; 
        time_latex = [perf_iRBBS(8) perf_ALM(8)]; 
        str_pi_iter = '%.1f/%.1f  & %.1f/%.1f\\\\\n'; 
        pi_iter_latex = [perf_iRBBS(14) perf_iRBBS(13) perf_ALM(14) perf_ALM(13)];
        if strcmp(art1,'Rom') 
            art1 = 'RJ';
        end
        if strcmp(art1,'Ham') 
            art1 = 'H';
        end
        if strcmp(art1,'Oth') 
            art1 = 'O';
        end
        if strcmp(art2,'Rom') 
            art2 = 'RJ';
        end
        if strcmp(art2,'Ham') 
            art2 = 'H';
        end
        if strcmp(art2,'Oth') 
            art2 = 'O';
        end
        if latex_show 
        fprintf(fid,strcat(str, str_PRW, str_nGrad, str_SK, str_time, str_pi_iter),...
            art1,art2,PRW_latex,nGrad_latex,SK_exp_log_latex,time_latex, pi_iter_latex); 

        end
    end
end
Perf_ALM = Perf_ALM./i_nd; 
Perf_iRBBS = Perf_iRBBS./i_nd; 

fprintf(fid, '\\cmidrule(l){2-11}  \n'); 
str = '%s &';
if Perf_iRBBS(10)>Perf_ALM(10)
    str_PRW = '\\textBF{%.5f} & %.5f &';
elseif Perf_iRBBS(10) < Perf_ALM(10)
    str_PRW = '%.5f & \\textBF{%.5f} &';
else
    str_PRW = '\\textBF{%.5f} & \\textBF{%.5f} &';
end
    
PRW_latex = [Perf_iRBBS(10) Perf_ALM(10)].*scale;
str_nGrad = '%.0f & %.0f &';
nGrad_latex = [Perf_iRBBS(5) Perf_ALM(5)];
str_SK = '%.0f/%.0f & %.0f/%.0f &';
SK_exp_log_latex = [Perf_iRBBS(6) Perf_iRBBS(7) Perf_ALM(6) Perf_ALM(7)];
str_time = '%.1f & %.1f &';
time_latex = [Perf_iRBBS(8) Perf_ALM(8)];
str_pi_iter = '%.1f/%.1f & %.1f/%.1f  \\\\\n';
pi_iter_latex = [Perf_iRBBS(14) Perf_iRBBS(13) Perf_ALM(14) Perf_ALM(13)];
fprintf(fid,strcat(str, str_PRW, str_nGrad, str_SK, str_time, str_pi_iter),...
   'AVG',PRW_latex,nGrad_latex,SK_exp_log_latex,time_latex, pi_iter_latex);


