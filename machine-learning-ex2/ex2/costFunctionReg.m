function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

tempJ = 0;

for i = 1:m,
  xi = X(i, :);
  xi = xi';
  z = theta' * xi;
  sigZ = sigmoid(z);
  tempJ = tempJ + -y(i) * log(sigZ) - (1 - y(i)) * log(1 - sigZ);
end;
tempJ = tempJ / m;

tempThetai = 0;
for i = 2:n,
  tempThetai = tempThetai + theta(i) ^ 2;
end;

regularizationCorrection = lambda * tempThetai / (2 * m);
  
J = tempJ + regularizationCorrection;

n = length(theta);
for j = 1:n,
  tempGrad = 0;
  for i = 1:m,
    xi = X(i, :);
    xi = xi';
    z = theta' * xi;
    sigZ = sigmoid(z);
    tempGrad = tempGrad + (sigZ - y(i)) * xi(j);
   

  end;

  regularization = 0;
  if j > 1 ,
    regularization = lambda * theta(j) / m;
  end;
    
  grad(j) = tempGrad / m + regularization;
end;




% =============================================================

end
