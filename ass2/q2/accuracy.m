function p = accuracy(X, Theta1, Theta2)
  [m n] = size(X);
  X = [ones(m, 1) X];
  a1 = sigmoid(X * Theta1');
  a1 = [ones(m, 1) a1];
  a2 = sigmoid(a1 * Theta2');
  [useless, p] = max(a2, [], 2);
end
