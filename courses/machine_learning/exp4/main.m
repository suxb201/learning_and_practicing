data_train = load('training_data.txt');
% data_train = data_train(1:9000, :);
data_test = load('test_data.txt');
len_test = length(data_test);
len_train = length(data_train);

num_attr = [3, 5, 4, 4, 3, 2, 3, 3];
cnt_correct = 0;
ans_num_y = [0, 0, 0, 0, 0];
for i = 1:len_test
    pridect = [0, 0, 0, 0, 0];
    for y = 0:4
        cnt_y = length(find(data_train(:, 9) == y));
        p_y = (cnt_y + 1) / (len_train + 5);
        log_x_y = 0;
        for j = 1:8
            cnt_x_y = length(find(data_train(:, 9) == y & data_train(:, j) == data_test(i, j)));
            p_x_y = (cnt_x_y + 1) / (cnt_y + num_attr(j));
            log_x_y = log_x_y + log(p_x_y);
        end
        pridect(y+1) = log(p_y) + log_x_y;
    end
    [~, y0] = max(pridect);
    ans_num_y(y0) = ans_num_y(y0) + 1;
    if data_test(i, 9) == y0 - 1
        cnt_correct = cnt_correct + 1;
    end
end

ans_rate = cnt_correct / len_test;
disp(ans_rate)
disp(ans_num_y)
