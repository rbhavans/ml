function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);


% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
htheta = X * theta;
tempTheta = theta;
tempTheta(1) = 0;
error = htheta - y;

regParam = lambda * sum(tempTheta .^ 2) / (2*m);

J = sum(error .^ 2) / (2*m) + regParam;

for j = 1:n
  grad(j) = sum(error .* X(:,j))/m + lambda * tempTheta(j) / m;
end
%DJ0 = sum((htheta - y) .* X(:,1)) / m;
%DJ1 = sum((htheta - y) .* X(:,2) + lambda * tempTheta) / m;
%
%grad = [DJ0, DJ1]







% =========================================================================

grad = grad(:);

end
