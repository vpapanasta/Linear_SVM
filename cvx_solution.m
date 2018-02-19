%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all;
% Load the data
cd ..
load(strcat(pwd, '\data', '\linear_svm.mat'));
cd src;

%% Training Stage
dim = size(X_train, 1);
X = X_train;
y = labels_train;
n = size(X, 1);
p = 2;

class_1 = zeros(dim, 2);
class_2 = zeros(dim, 2);

% Discreminate of points
for i = 1:dim
    if(y(i) == 1)
        class_1(i, :) = X(i, :);
    else
        class_2(i, :) = X(i, :);
    end
end

figure; 
plot(class_1(:,1), class_1(:,2), 'r.', class_2(:,1), class_2(:,2), 'g.');

tic
%%
cvx_begin
    variables w(p) b
    dual variables pi
    minimize ( 1/2*w'*w ) 
    subject to
    pi : y.*( X*w + b) >= 1;
cvx_end
time_ = toc;
% Find the first points in the two categories -1 and +1 (Vector supports)
vec_sup = find (y .*( X*w + b) <= 1 + eps ^.3) ;

%%
x1 = min(X(:));
y1 = (-b - (w(1)*x1))/w(2);
z1 = (1 - b - (w(1)*x1))/w(2);
zm1 = (-1 - b - (w(1)*x1))/w(2) ;
x2 = max(X(:));
y2 = (-b - (w(1)*x2))/w(2) ;
z2 = (1 - b - (w(1)*x2) )/w(2) ;
zm2 = (-1 - b - (w(1)*x2) )/w(2);

figure;
plot(class_1(:,1), class_1(:,2), 'r.', class_2(:,1), class_2(:,2), 'g.', [x1 x2], [y1 y2], 'b', [x1 x2], [z1 z2], ':k', [x1 x2], [zm1 zm2], ':k');
% figure; plot (X(vec_sup ,1), X(vec_sup, 2) , 'k', 'MarkerSize', 10); hold all;

%% Testing stage
dim = size(X_test, 1);
X = X_test;
y = labels_test;
n = size(X, 1);
p = 2;

class_1 = zeros(dim, 2);
class_2 = zeros(dim, 2);

% Discreminate of points
for i = 1:dim
    if(y(i) == 1)
        class_1(i, :) = X(i, :);
    else
        class_2(i, :) = X(i, :);
    end
end

figure; 
plot(class_1(:,1), class_1(:,2), 'r.', class_2(:,1), class_2(:,2), 'g.');

% Find the first points in the two categories -1 and +1 (Vector supports)
vec_sup = find (y .*( X*w + b) <= 1 + eps ^.3) ;

%%
x1 = min(X(:));
y1 = (-b - (w(1)*x1))/w(2);
z1 = (1 - b - (w(1)*x1))/w(2);
zm1 = (-1 - b - (w(1)*x1))/w(2) ;
x2 = max(X(:));
y2 = (-b - (w(1)*x2))/w(2) ;
z2 = (1 - b - (w(1)*x2) )/w(2) ;
zm2 = (-1 - b - (w(1)*x2) )/w(2);

figure;
plot(class_1(:,1), class_1(:,2), 'r.', class_2(:,1), class_2(:,2), 'g.', [x1 x2], [y1 y2], 'b', [x1 x2], [z1 z2], ':k', [x1 x2], [zm1 zm2], ':k');

% Determine the +1 and -1 margin lines
% +1 case
coeffs_n = polyfit([x1 x2], [zm1 zm2], 1);
a_p = coeffs_n(1);
b_p = coeffs_n(2);

% -1 case
coeffs_n = polyfit([x1 x2], [z1 z2], 1);
a_n = coeffs_n(1);
b_n = coeffs_n(2);

sum_pos = size(find(labels_test == 1), 1); % Count # of +1 elements
sum_neg = size(find(labels_test == -1), 1); % Count # of -1 elements
error_pos = 0;
error_neg = 0;

% Measure errorneous points 
for i = 1:dim
    if(y(i) == 1)
        if(a_n*X(i, 1) + b_n < X(i, 2))
            error_neg = error_neg + 1;
        end
    else
        if(a_p*X(i, 1) + b_p > X(i, 2))
            error_pos = error_pos + 1;
        end
    end
end

% Print the errors
fprintf('Error pos: %.3f \n', (error_pos/sum_pos)*100);
fprintf('Error neg: %.3f \n', (error_neg/sum_neg)*100);
fprintf('Average Error: %.3f \n', ((error_pos + error_neg)/dim)*100);
fprintf('Calculation time of w and b: %.6f secs \n', time_);