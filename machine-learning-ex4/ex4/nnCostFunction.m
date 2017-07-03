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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m,1) X];

z2 = a1 * Theta1';

a2 = sigmoid(z2);

a2 = [ones(m,1) a2];

a3 = sigmoid(a2 * Theta2');

h = a3;

ytemp = zeros(size(y,1),num_labels);

for i = 1:m
	ytemp(i,y(i,1))=1;
end

tTheta1 = [zeros(size(Theta1,1),1) Theta1(:,2:size(Theta1,2))];
tTheta2 = [zeros(size(Theta2,1),1) Theta2(:,2:size(Theta2,2))];


p = ( sum(sum(tTheta1.^2)) + sum(sum(tTheta2.^2)) );

J = 1/m * sum( sum(-ytemp .* log(h) - (1-ytemp) .* log(1-h),2) ) + lambda/2/m * p;









% for t = 1:m,
% 	a1 = X(t,:);
% 	a1 = [1 a1];
% 	% size(a1)
% 	% size(Theta1)
% 	z2 = Theta1 * a1';
% 	% size(z2)
% 	a2 = sigmoid(z2);
% 	a2 = [1;a2];

% 	z3 = Theta2 * a2;
% 	a3 = sigmoid(z3);

% 	% size(a3)
% 	% size(ytemp(t,:))


% 	delta3 = a3 - (ytemp(t,:))';

% 	z2 = [1;z2];

% 	delta2 = (Theta2' * delta3) .* sigmoidGradient(z2);

% 	delta2 = delta2(2:end);

% 	Theta2_grad = Theta2_grad + delta3 * a2';
% 	Theta1_grad = Theta1_grad + delta2 * a1;
% end


% Theta1_grad(:,1) = Theta1_grad(:,1) ./ m;
% Theta2_grad(:,1) = Theta2_grad(:,1) ./ m;


% Theta1_grad(:,2:end) = ( Theta1_grad(:,2:end) + (lambda * Theta1(:,2:end) ) ) ./m;
% Theta2_grad(:,2:end) = ( Theta2_grad(:,2:end) + (lambda * Theta2(:,2:end) ) ) ./m;










% ??????????



delta3 = a3 - ytemp;

% z2 = [ones(size(z2,1)) z2];

% size(a2)
% size(z2,1)
% size(delta3 * Theta2)
z2 = [ones(size(z2,1),1) z2];
% size(z2)
delta2 = delta3 * Theta2 .* sigmoidGradient(z2);

delta2 = delta2(:,2:end);


delta_cap1 = delta2' * a1;

delta_cap2 = delta3' * a2;

Theta1_grad(:,2:end) = 1/m * (delta_cap1(:,2:end) + lambda * Theta1(:,2:end));

Theta2_grad(:,2:end) = 1/m * (delta_cap2(:,2:end) + lambda * Theta2(:,2:end));


Theta1_grad(:,1) = delta_cap1(:,1) ./ m;

Theta2_grad(:,1) = delta_cap2(:,1) ./ m;




% Theta1_grad = 1/m * (delta_cap1 );

% Theta2_grad = 1/m * (delta_cap2 );


% Theta1_grad(:,1) = Theta1_grad(:,1) - 1/m * (delta_cap1(:,1));

% Theta2_grad(:,1) = Theta2_grad(:,1) - 1/m * (delta_cap2(:,1));

% ??????????



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
