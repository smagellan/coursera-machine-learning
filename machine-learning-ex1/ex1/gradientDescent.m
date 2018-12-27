function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
coeff = (alpha / m);
numThetas = size(theta);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    tt = theta';
    tx = X';

    %theta0 = theta(1);
    %theta1 = theta(2);
    %newTheta0 = theta0 - coeff * sum(((tt * tx)' - y) .* X(:, 1));
    %newTheta1 = theta1 - coeff * sum(((tt * tx)' - y) .* X(:, 2));

    newTheta = theta;
    for thetaIdx = 1:numThetas
        newTheta(thetaIdx) = theta(thetaIdx) - coeff * sum(((tt * tx)' - y) .* X(:, thetaIdx));
    end
    theta = newTheta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %theta = [newTheta0; newTheta1];

end
save 'J_history.dat' J_history
end

