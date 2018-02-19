%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gradient Descent Support Vector Machines (SVM)
clear all; clc; close all;

size_C = 100;
errors = zeros(1, size_C);
ii = 0;

%% Run the algorithm for several values of C in order to find the optimal value depending on the error
% Some GD parameters
w_t = zeros(3, 1); % w_1, w_2, b
for C = 1:size_C
    [~, error] = GD_svm_func(ii, C, 'for', w_t);
    ii = ii + 1;
    errors(ii) = error;
end
figure; plot(1:size_C, errors);
title('SVM classification Error vs C value'); xlabel('C value'); ylabel('Error %');
[jj, C_opt] = min(errors);
fprintf('Optimal value of C: %d \n\n', C_opt);


%% Run the algorithm for a single optimal value of C measuring execution time and error
% Some GD parameters
 w_t = zeros(3, 1); % w_1, w_2, b
[time_, ~] = GD_svm_func(ii, C_opt, 'single_opt', w_t);

fprintf('\nCalculation time of w and b: %.6f secs \n', time_);


%% Run the algorithm to check convergence
% Some GD parameters
w_t = zeros(3, 1); % w_1, w_2, b
[~, ~] = GD_svm_func(ii, C_opt, 'test_converge', w_t);