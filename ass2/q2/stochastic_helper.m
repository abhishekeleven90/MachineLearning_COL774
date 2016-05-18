function [J the1 the2] = stochastic_helper(the1,the2,isize, hsize, num_op, X, y, eta)

m = size(X, 1);

%%constructing the Y 2d matrix for all output
Y = eye(num_op)(y, :);

a1 = [ones(m, 1), X];

z2 = a1 * the1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1), a2];

z3 = a2 * the2';
a3 = sigmoid(z3);

% Cost (summation over all training example and all classes)
ccc = sum(sum((Y-a3).*(Y-a3)))/2;
%ccc = sum((Y-a3)*(Y-a3)')/2;
J = (1 / m) * sum(ccc);

for t = 1:m
  if mod(t,10) == 0
    %%disp(sprintf('t: %d ',t));
  end
  
  the1NoBias = the1(:, 2:end);
  the2NoBias = the2(:, 2:end);

	%%forward pass again and again, as theta is getting updated on each iteration 
    a1 = [1; X(t, :)'];
	z2 = the1 * a1;
	a2 = [1; sigmoid(z2)];

	z3 = the2 * a2;
	a3 = sigmoid(z3);

	d3 = a3 - Y(t, :)';
	d2 = (the2NoBias' * d3) .* sigmoidGradient(z2);

	del2 = (d3 * a2');
	del1 = (d2 * a1');
  
  the1 = the1 - eta*del1; %update theta in each iteration
  the2 = the2 - eta*del2; %update theta in each iteration
  
endfor

end