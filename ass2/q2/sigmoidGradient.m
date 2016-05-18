function g = sigmoidGradient(x)
    g = zeros(size(x));
    v = sigmoid(x);
    g = v.*(1-v);
end
