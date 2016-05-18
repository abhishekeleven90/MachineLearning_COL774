%%%%%%%%%%% HELP %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%To use put 
% type = 0: Linear line
% type = 1: Mesh
% type = 2: Contour
% put alpha(eta), epsilon also to check
% put waittime = 0.2 as in question, as asked
% also if you don't want to see the trail of animation put gayab = 1, else gayab = 0; 

%% clear init
clear ; close all; clc;
epsilon = 0.0001;
alpha = 2.5; %or eta for us
num_iters = 50;
type = 2; %%type 0:just line, type 1: mesh, type 2: contour
waittime = 0.1;
gayab = 0;


%%% Loading Data %%%
X = load('q1x.dat');
y = load('q1y.dat');
m = length(y);
[m n] = size(X);

% Normalize
for i=1:n
  mu = mean(X(:,i));
  sigma = std(X(:,i));
  X(:,i) = (X(:,i)-mu)/sigma;
end

%% not normalized y

%%% Intercept term %%%
X = [ones(m, 1) X];
[m n] = size(X);

% Init Theta 
theta = zeros(n, 1);
disp('Initial theta:')
disp(theta);
disp('Press to continue');


J_history = zeros(num_iters, 1); %%TODO: names of variables

%%if type 1 display mesh first %%
if(type==1 || type==2)
    disp('Mesh draw J(theta_0, theta_1)')

    %Grid
    theta0_vals = linspace(-1, 9, 130);
    theta1_vals = linspace(-1, 9, 130);

    J_vals = zeros(length(theta0_vals), length(theta1_vals));
    
    %Fill
    for i = 1:length(theta0_vals)
        for j = 1:length(theta1_vals)
            t = [theta0_vals(i); theta1_vals(j)];    
            J_vals(i,j) = calculateJ(X, y, t);
        end
    end
    
    %%Plot mesh
    %Have to transpose first
    J_vals = J_vals';
    
    if (type==1)
        %draw figure
        figure;
        surf(theta0_vals, theta1_vals, J_vals);hold on;
        mmt = sprintf('Mesh for alpha = %f',alpha);
        title(mmt);
        xlabel('\theta_0'); ylabel('\theta_1'); zlabel('J-Theta');
        hold on;
        disp('Mesh drawn. Press to continue to see animated convergence');
        hold on; %%very very imp will be used in convergence animation
        pause;
        pause;
    end
    
    
    
    if (type==2)
        figure;
        contour(theta0_vals, theta1_vals, J_vals);
        xlabel('theta0'); ylabel('theta1');
        ctt = sprintf('Contour for %f',alpha);
        title(ctt);
        hold on;
        disp('Contour drawn. Press to continue to see animated convergence');
        pause;
        pause;
    end

end




%%TODO: remove prints!!!

disp('Beginning gradient descent');
%% Actual gradient descent 
%%TODO: error bound!!!
for iter = 1:num_iters
  h = X * theta;
  theta = theta - (alpha * (1 / m) * (transpose(X) * (h - y)));
  
  %% mesh animate
  if (type==1)
    J_history(iter) = calculateJ(X, y, theta); %%TODO: bring inside!!
    clear ppp;
    ppp=plot3(theta(1),theta(2),J_history(iter),'rx','LineWidth',2); 
    hold on;
    pause(waittime);
    if (gayab==1)        
        delete(ppp);
    end
  end
  
  %%contour animate
  if (type==2)
    clear ppp;
    ppp=plot(theta(1),theta(2),'rx','LineWidth',2); 
    hold on;
    pause(waittime);
    if (gayab==1)
        delete(ppp);
    end
  end
end

disp(theta);
disp('Gradient descent done');

%%Linear plot
if (type==0)
    hold off;
    disp('Drawing line');
    figure;
    plot(X(:,2),y,'rx','LineWidth',2);
    hold on;
    xall = [min(X(:,2)):0.1:max(X(:,2))]';
    intercept = ones(1,size(xall,1))';
    disp(size(intercept));
    disp(size(xall));
    yall = [intercept xall] * theta;
    plot(xall,yall,'b','LineWidth',2);
    xlabel('x');
    ylabel('y');    
    ss = sprintf('Linear Regression Line using alpha = %f',alpha);
    title(ss);
    legend('Training Points','Linear regression line');
    hold off;
    disp('Line drawn');
end

disp('Program end');