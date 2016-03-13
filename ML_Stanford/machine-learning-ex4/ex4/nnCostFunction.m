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
[p1 q1]= size(Theta1);
[p2 q2]= size(Theta2);
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
%vv: lets first add bias unit for the input givens
X= [ones(m,1) X];
a2partial=sigmoid(X*Theta1.');
n = size(a2partial, 1);
a2=[ones(n, 1) a2partial];
a3=sigmoid(a2*Theta2.');

%vv: now we got htheta which is a3, we need to compute cost. in onevsall, we are calculating theta for every class individially ( theta0, theta1...theta400)
% and hence we have only one column of htheta and one column of y. In here we are calculating htheta for every class at once by providing one sample of input and calculate theta1, theta2 and provide next sample of input and calculate theta1 and theta2 and so on. 
%So while calculating the cost, we need to take all the classes into account unlike obevsall which only calculates one class at a time.
%a3 is 400x10, where every column represent one class, while calculating cost, Y should also be the size of 400x10 to consider all the classes.
%create y_matrix such that it should have value of 1 only at the position where it actual value lies.
%say you have y output as 5, your y_matrix should be like [ 0 0 0 0 1 0 0 0 0 0] i.e only  5 position should be 1 and rest should be 0.
% if they are not zero, they are contributing to increased theta. 

y_eye= eye(num_labels);
y_matrix= y_eye(y,:);
%y_eye(y,:) will take the rows (mentioned in y) of y_eye and save them in y_matrix

dummy=zeros(size(y_matrix));
dummy= ( (-y_matrix.*log(a3)) - ( (1-y_matrix).*log(1-a3) ) );
dummy=sum(dummy,2); % summing across columns which is summation for 1:K
dummy=sum(dummy,1);%summing across rows which is summation for 1:m
Reg_term_theta1_partial1= (lambda/(2*m))*sumsq(Theta1(:,2:q1)(:));
Reg_term_theta1_partial2= (lambda/(2*m))*sumsq(Theta2(:,2:q2)(:));
%In above equations, we dont want to add bias ter theta0 and hence removed first column
%summation in above term can be done in two steps: Step1: across K Step2: across J and then combine both. Instead we can just 
%add all the elements using inbuilt function sumsq(a((:))) which squares evry input and then summs up all of them
J=(1/m*dummy)+Reg_term_theta1_partial1+Reg_term_theta1_partial2;

small_delta_3= a3-y_matrix;
small_delta_2= small_delta_3*Theta2(:,2:end).*sigmoidGradient(X*Theta1.');
Big_Delta_1= small_delta_2.'*X;
Big_Delta_2=small_delta_3.'*a2;

Regularization_1= (lambda/m)*Theta1;
Regularization_1(:,1)=0;

Regularization_2= (lambda/m)*Theta2;
Regularization_2(:,1)=0;

Theta1_grad= 1/m*Big_Delta_1 + Regularization_1;
Theta2_grad= 1/m*Big_Delta_2 + Regularization_2;















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
