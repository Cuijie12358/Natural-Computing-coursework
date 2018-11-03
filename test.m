%% Test WCA
Npop = 60;
Nsr = 20;
dmax = 1e-5;
max_it = 100;
obj_f = @func;
const = @constraints;
Nr = 10; % The number of rerun
F = zeros(Nr);
for i = 1:Nr
    num_var = 7;
    lb = [-10 -10 -10 -10 -10 -10 -10];
    ub = [10 10 10 10 10 10 10];
    [Xopt, Fopt] = WCA_simple(obj_f,const,lb,ub,num_var,Npop, Nsr, dmax, max_it);
    F(i) = Fopt;
end

F_mean = mean(F);
F_std = std(F);
disp(['After ' , num2str(Nr), ' reruns : ']);
disp(['F_mean = ' , num2str(F_mean(1)), '    F_std = ',num2str(F_std(1))]);