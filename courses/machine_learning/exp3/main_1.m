% ------------- parameter --------------
lambda = 10;
epoch = 20;

title("lambda = " +num2str(lambda))
hold on;
x = load('ex3Logx.dat');
y = load('ex3Logy.dat');
m = length(x);
n = 28;
neg = find(y == 0);
pos = find(y == 1);

plot(x(pos, 1), x(pos, 2), '+');
hold on;
plot(x(neg, 1), x(neg, 2), 'o');
xlabel('u');
ylabel('v');
legend('y = 1', 'y = 0');

x = map_feature(x(:, 1), x(:, 2));
theta = zeros(n, 1);
g = @(x) 1.0 ./ (1.0 + exp(-x));

J = zeros(epoch, 1);
for i = 1:epoch
    h = g(x*theta);
    J(i) = m \ sum(-y.*log(h)-(1 - y).*log(1-h)) + (lambda / 2 / m) * norm(theta([2:end]))^2;
    G = (lambda / m) .* theta; % regu
    G(1) = 0;
    grad = m \ (x' * (h - y)) + G;
    L = (lambda / m) .* eye(n); % regu
    L(1) = 0;
    H = m \ (x' * diag(h) * diag(1-h) * x) + L;
    theta = theta - H \ grad;
end

% figure % J
% plot(0:epoch-1, J)

u = linspace(-1, 1.5, 200);
v = linspace(-1, 1.5, 200);
z = zeros(length(u), length(v));
for i = 1:length(u)
    for j = 1:length(v)
        z(i, j) = map_feature(u(i), v(j)) * theta;
    end
end

contour(u, v, z', [0, 0], 'LineWidth', 2)
