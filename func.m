%% Test object function
function obj = func(x)
%From paper Constrainted problem 1
obj = (x(1)-10)^2 + 5*(x(2)-12)^2 + x(3)^4 + 3 * (x(4)-11)^2 + 10 * x(5)^6 + 7 * x(6)^2 + x(7)^4 - 4*x(6)*x(7) - 10*x(6) - 8*x(7);