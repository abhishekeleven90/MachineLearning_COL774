%% init clear everything%%
clear; clc; close all;
stepsize = 0.3;
all_taus = [.8 .1 .3 2 10];
all_colors = ['g' 'c' 'y' 'k' 'm'];

%% load data, append col vector, size %% 
x = load('q3x.dat');
[m n] = size(x);
toadd = ones(m,1);
x = [toadd x];
[m n] = size(x);
y = load('q3y.dat');
xorig = x(:,2);

%% fit linear using normal
xmin = min(xorig);
xmax = max(xorig);
xall = [xmin:stepsize:xmax];
the = (transpose(x)*x)^-1*transpose(x)*y; %theta
yall = the(1)+the(2)*xall;
disp('final theta for linear:');
disp(the);

%% plot linear+training points with data points
figure;hold on;plot(xorig,y,'xb','LineWidth',2);
plot(xall,yall,'r','LineWidth',2);
title('Linear Regression using Normal Equations');
xlabel('x');
ylabel('y');
legend('Training Points','Linear Regression Line');
hold off;


loopsize = size(all_taus,2);

for p=1:loopsize %for each tau
  curr_color = all_colors(p);
  curr_tau = all_taus(p);
  plots(p) = draw_lwlr(x,y,stepsize,curr_tau,curr_color);
end