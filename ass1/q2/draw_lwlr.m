function fig = draw_lwlr(x,y,stepsize,tau,color)

[m n] =size(x);
xorig = x(:,2);
xmin = min(xorig);
xmax = max(xorig);
xall = [xmin:stepsize:xmax];
xallsize = size(xall,2); 


for k=1:xallsize %%% for each query point x in xall
  weight = zeros(m,m); %%weighted matrix
  for j=1:m %%% for each training example
    num = (xall(k)-x(j,2));
    num = num^2;
    den = (2*tau^2);
    weight(j,j)=exp(-(num/den));
  end
  theta = pinv(transpose(x)*weight*x)*transpose(x)*weight*y;
  yall(k) = theta(1)+theta(2)*xall(k) ;
end
disp(sprintf('theta for linear weighted with tau = %f',tau));
disp(theta);
pause;


figure;hold on;plot(xorig, y,'xb','LineWidth',2);
fig = plot(xall,yall,color,'LineWidth',2);
title(sprintf('Locally Weighted Linear Regression Fit for tau = %f',tau));
xlabel('x');
ylabel('y');
legend('Training Points',sprintf('tau=%f',tau));
hold off;