function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
thetazero=theta(1,:);
thetaone=theta(2,:);
htheta(:,1)=X*theta;
i=1;
dummy1=0;
dummy2=0;
for i=1:m
dummy1=dummy1+ ( ( htheta(i,1)-y(i,1) )*X(i,1) );
dummy2=dummy2+ ( ( htheta(i,1)-y(i,1) )*X(i,2) );
endfor

thetazero=thetazero-( (alpha/m)*dummy1);
thetaone=thetaone-( (alpha/m)*dummy2);
theta=[thetazero;thetaone];




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
