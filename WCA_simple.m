function [Xopt,Fopt] = WCA_simple(obj_f, const, lb, ub, num_var, Npop, Nsr, dmax, max_it, numq)
%% Info
% This is a function to find a minimum solution!
% author {Yu, Wei(Cyrus)} - This algorithm is only for Natural Computing
% Cousework.
%Inputs:
%obj_f :    Objective function
%const :    Constraints
%lb:        Lower bound
%ub:        Upper bound
%num_var :  Number of variables
%Npop :     Population size
%Nsr :      Number of rivers + sea
%dmax :     Evaporation condition
%max_it :   Maximum of iterations
%numq :     The function and constraints of a specific problem 
%Outputs:   
%Xopt :     Optimum solution
%Fopt :     Fitness of solution
%% Code
% Initial population
raindrop = zeros(1, num_var);
pop = repmat(raindrop, Npop, 1);
cost = zeros(Npop, 1);
for i = 1:Npop 
    pop(i,:) = lb + (ub - lb).*rand(1,num_var);
    while const(pop(i,:),numq) == 0
        pop(i,:) = lb + (ub - lb).*rand(1,num_var);
    end 
    cost(i) = obj_f(pop(i,:),numq);
end

%sea and rivers
Nrds = Npop - Nsr;
[~, ind] = sort(cost);
sea = pop(ind(1),:);
rivers = repmat(raindrop, Nsr-1, 1);
for i = 1:Nsr-1
    rivers(i,:) = pop(ind(1+i),:);
end

%others (streams / raindrops)
raindrops = repmat(raindrop, Nrds, 1);
for i = 1:Nrds
    raindrops(i,:) = pop(ind(Nsr+i),:);
end

%NSn
NSn = zeros(Nsr, 1);
sum_of_Csr = sum(obj_f(sea,numq));
for i=1:Nsr-1
    sum_of_Csr = sum_of_Csr + obj_f(rivers(i,:),numq);
end
for i = 1:Nsr
    if i == 1 
        NSn(i) = round(abs(obj_f(sea,numq)/sum_of_Csr) * Nrds);
    else
        NSn(i) = round(abs(obj_f(rivers(i-1,:),numq)/sum_of_Csr) * Nrds);
    end
end

if sum(NSn) > Nrds
    ext = sum(NSn) - Nrds;
    for i = 1:Nsr
        if NSn(i) ~= 0
            NSn(i) = NSn(i) - 1;
            ext = ext - 1;
            if ext == 0
                break
            end
        end
    end
end
% loop 
C = 2;
F_pre = zeros(max_it,1);
for i=1:max_it
    % stream to river
    for j = 1:Nsr
        for k = 1:NSn(j)
            indx = sum(NSn(1:j-1)) + k;
            tmp = raindrops(indx,:);
            if j > 1
                raindrops(indx,:) = raindrops(indx,:) + C .* rand(1,num_var) .* (rivers(j-1,:) - raindrops(indx,:));
                if const(raindrops(indx,:),numq) == 0
                    raindrops(indx,:) = tmp;
                end
            else
                raindrops(indx,:) = raindrops(indx,:) + C .* rand(1,num_var) .* (sea - raindrops(indx,:));
                if const(raindrops(indx,:),numq) == 0
                    raindrops(indx,:) = tmp;
                end
            end
            raindrops(indx,:) = min(raindrops(indx,:),ub);
            raindrops(indx,:) = max(raindrops(indx,:),lb);
            cost_ra = obj_f(raindrops(indx,:),numq);
            cost_s = obj_f(sea,numq);
            if j > 1
                cost_r = obj_f(rivers(j-1,:),numq);
                if cost_ra < cost_r
                    n_river = raindrops(indx,:);
                    raindrops(indx,:) = rivers(j-1,:);
                    rivers(j-1,:) = n_river;
                    cost_r = obj_f(rivers(j-1,:),numq);
                    if cost_r < cost_s
                        n_sea = rivers(j-1,:);
                        rivers(j-1,:) = sea;
                        sea = n_sea;
                    end
                end
            else 
                if cost_ra < cost_s
                    n_sea = raindrops(indx,:);
                    raindrops(indx,:) = sea;
                    sea = n_sea;
                end
            end
        end
    end
    % river to sea
    for j = 1:Nsr-1
        tmp = rivers(j,:);
        rivers(j,:) = rivers(j,:) + C .* rand(1,num_var) .* (sea - rivers(j,:));
        if const(rivers(j,:),numq) == 0
            rivers(j,:) = tmp;
        end
        rivers(j,:) = min(rivers(j,:),ub);
        rivers(j,:) = max(rivers(j,:),lb);
        cost_r = obj_f(rivers(j,:),numq);
        cost_s = obj_f(sea,numq);
        if cost_r < cost_s
            n_sea = rivers(j,:);
            rivers(j,:) = sea;
            sea = n_sea;
        end
    end
    % evaporation condition
    for j = 1:Nsr-1
        if abs(sea - rivers(j,:)) < dmax
            for k=1:NSn(j)
                indx = k + sum(NSn(1:j));
                raindrops(indx,:) = lb + rand(1,num_var).*(ub-lb);
                while const(raindrops(indx,:),numq) == 0
                    raindrops(indx,:) = lb + (ub - lb).*rand(1,num_var);
                end 
            end
        end
    end
    dmax = dmax - (dmax/max_it);
    
    cost_s = obj_f(sea,numq);
    disp(['No.' , num2str(i), '    F= ',num2str(cost_s)]);
    F_pre(i) = cost_s;
end
%% Result
plot(F_pre,'LineWidth',1);
xlabel('Number of Iterations');
ylabel('Function Values');
Xopt = sea;
Fopt = cost_s;
hold on
end
    
