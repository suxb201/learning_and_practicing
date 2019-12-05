x = load("data2/ex2x.dat");
y = load("data2/ex2y.dat");
n = length(x);
x = [ones(n, 1), x];
stds = std(x);
mu = mean(x);
x(:, 2) = (x(:, 2) - mu(2)) ./ stds(2);
x(:, 3) = (x(:, 3) - mu(3)) ./ stds(3);

% check 1: Plot the Data
pos = find(y == 1);
neg = find(y == 0);
% figure('Name', 'Exam 1 score')
% plot(x(pos, 2), x(pos, 3), '+')
% hold on
% plot(x(neg, 2), x(neg, 3), 'o')

% check 2: Gradient descent
epsilon = 1e-6;
alpha = 0.02;
theta = double(zeros(1, 3));
theta = [0.01, 0.01, 0.01];

h = @(theta) 1.0 ./ (1.0 + exp(-x*theta.'));
L = @(theta) - mean(y.*log(h(theta))+(1 - y).*log(1-h(theta)));
dL = @(theta) mean((h(theta) - y).*x);
predict = @(v1, v2, theta) 1.0 / (1.0 + exp(-[1, (v1 - mu(2)) / stds(2), (v2 - mu(3)) / stds(3)]*theta.'));

d = Inf;
iter = 0;
list_L = [L(theta)];
while d > epsilon
    last_L = L(theta);
%   theta = theta - alpha * dL(theta);
    theta = theta - dL(theta)/H(theta, x, n);
   
    now_L = L(theta);
    d = abs(last_L-now_L);
    iter = iter + 1;
    list_L = [list_L; [now_L]];
end

disp('iter')
disp(iter) % q1
disp('theta')
disp(theta) % q2
figure('Name', 'L')
plot(list_L) % q3
figure('Name', 'The decision boundary')
x1 = x(:, 2);
x2 = x(:, 3);
plot(x(pos, 2)*stds(2)+mu(2), x(pos, 3)*stds(3)+mu(3), '+')
hold on
plot(x(neg, 2)*stds(2)+mu(2), x(neg, 3)*stds(3)+mu(3), 'o')
hold on
plot(x1*stds(2)+mu(2), (-x(:, 1:2) * theta(1:2).' / theta(3))*stds(3)+mu(3))
disp('predict')
disp(1-predict(20, 80, theta)) % q4

function res = H(theta, x, n)
    res = zeros(3, 3);
    for i = 1:n
        tmp = 1.0 ./ (1.0 + exp(-x(i, :)*theta.'));
        tmp = tmp * (1 - tmp) * x(i, :).' * x(i, :);
        res = res + tmp;
    end
    res = res ./ n;
end