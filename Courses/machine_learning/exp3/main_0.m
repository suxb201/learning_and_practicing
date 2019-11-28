% ------------- parameter --------------
lambda = 10;
title("lambda = " +num2str(lambda))

hold on;

x = load('ex3Linx.dat');
y = load('ex3Liny.dat');
m = length(x);

% point dots
plot(x, y, 'r.', 'Markersize', 15);
hold on;

func_x = @(m, x) [ones(m, 1), x, x.^2, x.^3, x.^4, x.^5];

x = func_x(m, x);
L = lambda .* eye(6);
L(1) = 0;
theta = (x' * x + L) \ x' * y; %  \ ~ inve

the_x_line = (linspace(-1, 1, 1000))';
the_x = func_x(1000, the_x_line);

the_y = the_x * theta;

plot(the_x_line, the_y);
