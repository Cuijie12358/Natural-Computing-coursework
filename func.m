%% Test object function
function obj = func(x,nq)
%From paper Constrainted problem 1
if nq == 1
    obj = (x(1)-10)^2 + 5*(x(2)-12)^2 + x(3)^4 + 3 * (x(4)-11)^2 + 10 * x(5)^6 + 7 * x(6)^2 + x(7)^4 - 4*x(6)*x(7) - 10*x(6) - 8*x(7);
elseif nq == 2
    obj = 5.3578547*x(3)^2 + 0.8356891*x(1)*x(5) + 37.293239*x(1) - 40729.141;
elseif nq == 3
    obj = 100*(x(2)-x(1)^2)^2 + (1-x(1))^2 + 90*(x(4)-x(3)^2)^2 + (1-x(3))^2 + 10.1*((x(2)-1)^2 + (x(4)-1)^2) + 19.8*(x(2)-1)*(x(4)-1);
end
