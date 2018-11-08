function [Xopt,Fopt] = PSO_simple(obj_f, const, lb, ub, num_var, p_num, omi, a1, a2,max_it, numq)
%% Info 
% This is a function to find a minimum solution!
% author {Yu, Wei(Cyrus) && Jie Cui} - This algorithm is only for Natural Computing
% Cousework.
%Inputs:
%obj_f :    Objective function
%const :    Constraints
%lb:        Lower bound
%ub:        Upper bound
%num_var :  Number of variables
%p_num :    Population size
%omi :      Omega(0.1 ~ 0.9)
%a1 :       Alpha1
%a2 :       Alpha2 (a1 + a2 = 4)
%max_it :   Maximum of iterations
%numq :     The function and constraints of a specific problem 
%Outputs:   
%Xopt :     Optimum solution
%Fopt :     Fitness of solution
%% Code
% Initialization
x = zeros(1, num_var);
X = repmat(x, p_num, 1);
V = zeros(p_num, num_var);
cost = zeros(p_num, 1);
g_glo = Inf;
g = zeros(1,num_var);
p_loc = zeros(p_num,1);
p_loc(:,1) = Inf;
p = zeros(p_num,num_var);

for i = 1:p_num 
    X(i,:) = lb + (ub - lb).*rand(1,num_var);
    while const(X(i,:),numq) == 0
        X(i,:) = lb + (ub - lb).*rand(1,num_var);
    end 
end

%loop
F_pre = zeros(max_it,1);
for i = 1:max_it
    for k = 1:p_num
        cost(k) = obj_f(X(k,:),numq);
        if cost(k) < p_loc(k)
            p(k,:) = X(k,:);
            p_loc(k) = cost(k);
        end
    end
    [val, ind] = sort(cost);
    if val(1) < g_glo
        g = X(ind(1),:);
        g_glo = val(1);
    end
    
    %update
    for k = 1:p_num
        r1 = rand();
        r2 = rand();
        tmpx = X(k,:);
        tmpv = V(k,:);
        V(k,:) = omi * V(k,:) + a1 * r1 * (p(k,:) - X(k,:)) + a2 * r2 * (g - X(k,:));
        X(k,:) = X(k,:) + V(k,:);
        if or(const(X(k,:),numq) == 0 , or(any(X(k,:)>ub), any(X(k,:)<lb)))
            X(k,:) = tmpx;
            V(k,:) = tmpv;
       end
    end
    
    %result
    disp(['No.' , num2str(i), '    F= ',num2str(g_glo)]);
    F_pre(i) = g_glo;
end
%% Plot
plot(F_pre,'LineWidth',1);
Xopt = g;
Fopt = g_glo;
legend('WCA','PSO')
hold off
end
