function j = calculateJ(x, y, theta)

m = length(y);
h = x*theta;
sqErr = (h-y).^2;
j = (1/(2*m))*sum(sqErr);

end
