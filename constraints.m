%% Function Constraints
function const = constraints(x)
%From paper Constrained problem 1
test = [127 - 2*x(1)^2 - 3*x(2)^4 - x(3) - 4*x(4)^2 - 5*x(5);
           282 - 7*x(1) - 3*x(2) - 10*x(3)^2 - x(4) + x(5);
           196 - 23*x(1) - x(2)^2 - 6*x(6)^2 + 8*x(7);
           -4*x(1)^2 - x(2)^2 + 3*x(1)*x(2) - 2*x(3)^2 - 5*x(6) + 11*x(7)];
if all(test >= 0)
    const = 1;
else
    const = 0;
end