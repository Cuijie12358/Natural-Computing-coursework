%% Function Constraints
function const = constraints(x,nq)
%From paper Constrained problem 1
if nq == 1  
    test = [127 - 2*x(1)^2 - 3*x(2)^4 - x(3) - 4*x(4)^2 - 5*x(5);
               282 - 7*x(1) - 3*x(2) - 10*x(3)^2 - x(4) + x(5);
               196 - 23*x(1) - x(2)^2 - 6*x(6)^2 + 8*x(7);
               -4*x(1)^2 - x(2)^2 + 3*x(1)*x(2) - 2*x(3)^2 - 5*x(6) + 11*x(7)];
    if all(test >= 0)
        const = 1;
    else
        const = 0;
    end
elseif nq == 2
    test = [85.334407 + 0.0056858*x(2)*x(5) + 0.0006262*x(1)*x(4) - 0.0022053*x(3)*x(5) - 92;
            -85.334407 - 0.0056858*x(2)*x(5) - 0.0006262*x(1)*x(4) - 0.0022053*x(3)*x(5);
            80.51249 + 0.0071317*x(2)*x(5) + 0.0029955*x(1)*x(2) + 0.0021813*x(3)^2 - 110;
            -80.51249 - 0.0071317*x(2)*x(5) - 0.0029955*x(1)*x(2) - 0.0021813*x(3)^2 + 90;
            9.300961 + 0.0047026*x(3)*x(5) + 0.0012547*x(1)*x(3) + 0.0019085*x(3)*x(4) - 25;
            -9.300961 - 0.0047026*x(3)*x(5) - 0.0012547*x(1)*x(3) - 0.0019085*x(3)*x(4) + 20];
    if all(test <= 0)
        const = 1;
    else
        const = 0;
    end
elseif nq == 3
    const = 1;
end