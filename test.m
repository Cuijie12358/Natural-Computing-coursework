%% Test WCA & PSO
numq = 3;
obj_f = @func;
const = @constraints;
max_it = 100;
%paras for WCA
Npop = 50;
Nsr = 10;
dmax = 1e-3;
%paras for PSO
p_num = 100;
omi = 0.7;
a1 = 2;
a2 = 2;

Nr = 25; % The number of rerun
F_w = zeros(1,Nr);
F_p = zeros(1,Nr);
for i = 1:Nr
    if numq == 1
        num_var = 7;
        lb = [-10 -10 -10 -10 -10 -10 -10];
        ub = [10 10 10 10 10 10 10];
    elseif numq == 2
        num_var = 5;
        lb = [78 33 27 27 27];
        ub = [102 45 45 45 45];
    elseif numq == 3
        num_var = 4;
        lb = [-5 -5 -5 -5];
        ub = [5 5 5 5];
    end
    [Xoptw, Foptw] = WCA_simple(obj_f,const,lb,ub,num_var,Npop, Nsr, dmax, max_it, numq);
    [Xoptp, Foptp] = PSO_simple(obj_f, const, lb, ub, num_var, p_num, omi, a1, a2,max_it, numq);
    F_w(i) = Foptw;
    F_p(i) = Foptp;
end

Fw_mean = mean(F_w);
Fw_std = std(F_w);
Fw_b = min(F_w);
Fw_w = max(F_w);
Fp_mean = mean(F_p);
Fp_std = std(F_p);
Fp_b = min(F_p);
Fp_w = max(F_p);
disp(['After ' , num2str(Nr), ' reruns : ']);
disp(['Fw_mean = ' , num2str(Fw_mean(1)), '    Fw_std = ',num2str(Fw_std(1)), '    Fw_best = ',num2str(Fw_b), '    Fw_worse = ',num2str(Fw_w)]);
disp(['Fp_mean = ' , num2str(Fp_mean(1)), '    Fp_std = ',num2str(Fp_std(1)), '    Fp_best = ',num2str(Fp_b), '    Fp_worse = ',num2str(Fp_w)]);