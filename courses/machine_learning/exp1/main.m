% data
x=load('ex1_2x.dat');
y=load('ex1_2y.dat');
m = length(x);
x = [ones(m,  1) ,  x]; 
global x m y;
disp(m);
sigma = std(x);
mu = mean(x);
x(:, 2) = (x(:, 2) - mu(2)) ./ sigma(2);
x(:, 3) = (x(:, 3) - mu(3)) ./ sigma(3);

% parameter
list_lr = [0.001, 0.01, 0.1,0.5,1.3]; % lr
iteration = 50;
lr_number = length(list_lr);
ans_j = zeros(iteration,lr_number);
legend_str = [];

% train
for th_lr = 1:lr_number
    lr = list_lr(th_lr);
    theta = zeros(3,1);
    for iter = 1:iteration
        ans_j(iter, th_lr) = loss(theta);
        theta = theta-lr/m*(sum(x.*(x*theta-y)).');
    end
    legend_str{th_lr} = ['lr= ',num2str(lr)]; %#ok<SAGROW>
    disp(['lr= ',num2str(lr),' theta: ', num2str(theta.')])
    disp(['price: ', num2str([1, (1650-mu(2))/sigma(2),(3-mu(3))/sigma(3)] * theta)])
end

% figure begin
figure ;
plot(0:iteration-1, ans_j);
xlabel('Number of iterations');
ylabel('Cost J');
legend(legend_str);

% loss function
function res = loss(theta)
    global x m y;
    t = x*theta;
    res = (x*theta-y)'*(x*theta-y)/(m*2);
end