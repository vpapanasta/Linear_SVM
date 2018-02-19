function [time_, error] = GD_svm_func(ii, C, flag, w_t)

    % Load the data
    cd ..
    load(strcat(pwd, '\data', '\linear_svm.mat'));
    cd src;

    x = X_train;
    y = labels_train;

    % Padd data
    x = [x, ones(size(x, 1), 1)];
    N = size(X_train, 1);  

    t = 0;
    if(strcmp(flag, 'for') || strcmp(flag, 'single_opt'))
        iterations = 15;
    elseif(strcmp(flag, 'test_converge'))
        iterations = 50;
        WW = zeros(3, iterations);
    end

    if(strcmp(flag, 'single_opt'))
        tic;
        C = 27; 
    end

    lamda = 1/(N*C); %

    % The Pegasus algorithm
    for iter = 1:iterations % Fixed number of iterations
        for j = 1:N % For the data length

            t = t + 1;
            % Decreasing for each iteration, step size (updating)
            htta = 1/(lamda*t); 
            % Choose a direction
            if(y(j)*(x(j, :)*w_t) < 1)
               w_t = (1 - htta*lamda)*w_t + htta*(y(j)*x(j, :))';
            else
               w_t = (1 - htta*lamda)*w_t;
            end

        end
        
        if(strcmp(flag, 'test_converge'))
            WW(: ,iter) = w_t;
        end
        
    end
    
    if(strcmp(flag, 'test_converge'))
        figure; plot(1:iterations, WW(1, :), 'r', 1:iterations, WW(2, :), 'g', 1:iterations, WW(3, :), 'b');
        title('Convergence of w & b'); xlabel('Values of w & b'); ylabel('Iterations'); legend('w(1)', 'w(2)', 'b');
    end

    w = w_t(1:2); b = w_t(3);
    if(strcmp(flag, 'single_opt'))
        time_ = toc;
    else
        time_ = 0;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    
    
    % figure; 
    % plot(class_1(:,1), class_1(:,2), 'r.', class_2(:,1), class_2(:,2), 'g.');

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

    if(strcmp(flag, 'single_opt'))
        figure;
        plot(class_1(:,1), class_1(:,2), 'r.', class_2(:,1), class_2(:,2), 'g.', [x1 x2], [y1 y2], 'b', [x1 x2], [z1 z2], ':k', [x1 x2], [zm1 zm2], ':k');
        title('SVM dataset classification');
    end

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

    %     Print the errors
    if(strcmp(flag, 'single_opt'))
        fprintf('Error pos: %.3f%%  \n', (error_pos/sum_pos)*100);
        fprintf('Error neg: %.3f%% \n', (error_neg/sum_neg)*100);
        fprintf('Average Error: %.3f%% \n', ((error_pos + error_neg)/dim)*100);
    end
        
    error = ((error_pos + error_neg)/dim)*100;
    
end