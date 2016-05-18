%%% clear previous work %%%%
clc; 
clear;
close all;

%%% loading data %%%%
y = load('q2y.dat');
x = load('q2x.dat');

%%% figuring data to plot %%%%
[m n] = size(x);
x = [ones(m,1) x];
[m n] = size(x); %%updating again for n
x1 = x(:,2); %%useless after new update
x2 = x(:,3);

%%% plotting indvidual points %%%%
pos = x(find(y==1),:);
neg = x(find(y==0),:);

figure;
plot(pos(:,2),pos(:,3),'r.','LineWidth',3);
hold on;
plot(neg(:,2),neg(:,3),'bx','LineWidth',3);

%%% adding labels to the only figure
title('Question 3: Logistic Regression using Normal Method - RED(1), BLUE(0)');
ylabel('x2');
xlabel('x1');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% theta calculate here %%%%%%%%%%%%%%%%%%%%%

stop = 0; %%stops when stop = 1
the = zeros(n,1); % assuming the inrecept x is already added!
the_old = zeros(n,1); %useless, for now, will see if can improve the stopping criteria

%stopping criteria could be iterations or error bound
%%using errors bound here
while(stop == 0)

  %% init g and H
  g=zeros(n,1); %will have to be constructed at each step
  H=zeros(n,n); %same here, calculate again and again
  
  done=1; %% if done still 1 after loop we hhave converged we need to stop
  
  %% due to the sigma summation step
  for k=1:m %no. of training examples
    
    %% sigmoid calcualte %%
    temp = x(k,:)*the;
    sig = 1./(1+exp(-temp)); %%sigmoid%%
    
    %% g and H update for each
    g_update = transpose(x(k,:))*(y(k)-sig); 
    h_update = sig*(1-sig)*transpose(x(k,:))*x(k,:); 
    g = g+g_update;
    H = H-h_update;
 
  end
  the_old = the;
  the_update = inv(H)*g;
  the = the - the_update;
  
  %% if all the values in gradient < 0.1, 
  %% done flag is not touched, else we iterate again.
  %% check for all features -- > all the gradient values
  for p=1:n
  %%tried 0.001, 0.001 all same almost answer all giving same exact answer
    if(abs(g(p)) > 0.01) 
        done=0;
    end
  end
    
  %%stop converence is checked here
  if(done==1)
    stop=1;
  end
  
end
disp('Theta: '); disp(the);
disp('Gradient: '); disp(g);
disp('Hessian: '); disp(H);

%%%%%%%%%%%%%%%%%%%%%% theta calcultae end %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% PLOT LINE %%%%
min_x1 = min(x(:,2)); 
max_x1 = max(x(:,2));
stepsize = 0.5;
xquery = min_x1:stepsize:max_x1;
plot(xquery, -the(2)/the(3)*xquery-the(1)/the(3),'k','LineWidth', 3); %against x1 we plot x2/y, plot(x,y)!!

%% LEGEND %%%%
legend('Positive','Negative','Logistic Reegression Line');