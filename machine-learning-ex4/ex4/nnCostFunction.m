function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

workingX = [ones(m , 1) , X];
cost = 0;


for i = 1:m,
  yivec = zeros(num_labels, 1);
  yivec(y(i)) = 1;

  zi2 = workingX(i,:) * Theta1';
  a2 = sigmoid(zi2);
  biaseda2 = [1, a2];
  zi3 = biaseda2 * Theta2';
  a3 = sigmoid(zi3);
  
  for k = 1 :  num_labels,
    cost = cost + (yivec(k) * log(a3(k)) + (1-yivec(k)) * log(1 - a3(k)));
  end;
end;

J = -cost/m;

temp1 = Theta1;
temp1(:, 1) = zeros(hidden_layer_size, 1);
temp2 = Theta2;
temp2(:, 1) = zeros(num_labels, 1);

J = J + lambda * (sum(sum(temp1 .^ 2)) + sum(sum(temp2 .^2))) / (2*m);


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

DELTA_2 = zeros(num_labels, hidden_layer_size + 1);
DELTA_1 = zeros(hidden_layer_size, input_layer_size + 1);
for i = 1:m,
  yivec = zeros(num_labels, 1);
  yivec(y(i)) = 1;

  zi2 = workingX(i,:) * Theta1'; % Zi2 is  1x25
  a2 = sigmoid(zi2); % a2 at this time is 1x25
  biasedzi2 = [1, zi2]'; %biasedzi2 is 1 x 26
  biaseda2 = [1, a2]; %biaseda2 is 1 x 26
  zi3 = biaseda2 * Theta2'; %zi3 is 1 x 10
  a3 = sigmoid(zi3); % a3 is 1 x 10
  a3 = a3';
  biaseda2 = biaseda2';
  
  % though a3 and y are row vectors, I am making delta3 a column vector
  % to work well with the formula given in the notes.
  delta3 = zeros(num_labels, 1); % delta3 is 10x1

  for k = 1 :  num_labels,
    delta3(k) = a3(k) - yivec(k);
  end;
 
  delta2 = (Theta2' * delta3) .* biaseda2 .* (1 - biaseda2);

%  size(delta3)
%  size(biaseda2)

  DELTA_2 = DELTA_2 + delta3 * biaseda2';

  DELTA_2(:,2:end) = DELTA_2(:,2:end) + lambda * Theta2(:,2:end) / m;

  DELTA_1 = DELTA_1 + delta2(2:end) * workingX(i,:);

  DELTA_1(:,2:end) = DELTA_1(:,2:end) + lambda * Theta1(:,2:end) / m;


end;



Theta2_grad = DELTA_2/m;
Theta1_grad = DELTA_1/m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
