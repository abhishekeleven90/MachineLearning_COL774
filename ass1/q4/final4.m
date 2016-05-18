%% clear stuff %%
clc;
clear;
close;

%% load original data %% 
x = load('q4x.dat');
txt = textread('q4y.dat', '%s', 'delimiter', '\n','whitespace', '');
[m n] = size(x);

%% normalize x - though was not needed %%
x(:,1) = (x(:,1) - mean(x(:,1)))/std(x(:,1));
x(:,2) = (x(:,2) - mean(x(:,2)))/std(x(:,2));

%% 100*1 boolean vectors for Canada:0 and Alaska:1 %%
y0 = strcmp(txt,'Canada'); 
y0_indices = find(y0==1);

y1 = 1-y0; %%complement of y1 boolean matrix
y1_indices = find(y1==1); 

canada = x(y0_indices,:); % [x1 x2] values for which Canada is true, used in plotting
alaska = x(y1_indices,:); % [x1 x2] values for which Alaska is true, used in plotting

%% y matrices %%
n1 = sum(y1); %% m = n0+n1;
n0 = sum(y0);
y0_t = transpose(y0);
y1_t = transpose(y1);

%% prob of y Alaska %%
phi = n1/m;
 
%% means for Canada 0 and Alaska 1
mu0 = transpose(y0_t*x)/n0;
mu1 = transpose(y1_t*x)/n1;

disp('phi');
disp(phi);
disp('mu0');
disp(mu0);
disp('mu1');
disp(mu1);
disp('');
disp('Press to continue');
pause;

%%% sigma, sigma0, sigma1 calc %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma0 = zeros(2);
sigma = zeros(2);
sigma1 = zeros(2);
for i=1:m
  
    %% sigma calc %%
    curr_x = x(i,:);
    curr_y0 = y0(i);
    curr_y1 = y1(i);
    
    interm = curr_x - transpose(mu1)*curr_y1 - transpose(mu0)*curr_y0; %%whichever is 1!!
    sigma_update = transpose(interm)*interm;
    sigma = sigma + sigma_update;
    
    %% sig1 and sig2 calc %%
    tmp0 = curr_x - transpose(mu0)*curr_y0;
    tmp1 = curr_x - transpose(mu1)*curr_y1; %%whichever is 1!!
    sigma0_update = curr_y0.*(transpose(tmp0)*tmp0);
    sigma0 = sigma0 + sigma0_update;
    sigma1_update  = curr_y1.*(transpose(tmp1)*tmp1);
    sigma1 = sigma1 + sigma1_update;
    
end

sigma0 = sigma0/n0;
sigma = sigma/(n0+n1);
sigma1 = sigma1/n1;

disp('sigma');
disp(sigma);
disp('sigma0');
disp(sigma0);
disp('sigma1');
disp(sigma1);
disp('');
disp('Press to continue');
pause;

%%% all inverses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inv0 = pinv(sigma0);
invsame = pinv(sigma);
inv1 = pinv(sigma1);

disp('inv');
disp(invsame);
disp('inv0');
disp(inv0);
disp('inv1');
disp(inv1);
disp('');
disp('Press to draw plots');
pause;


%%%%% SIGMA INVERSE %%%%
%% a b
%% c d
%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%% PLOTTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plotting individual points %%

plot(alaska(:,1),alaska(:,2),'g+','LineWidth',2);
hold on;
plot(canada(:,1),canada(:,2),'r*','LineWidth',2);


%%%%%% LDA variables and plot %%%%%%%%%%%%%%%%%%%%%%%%
mu01 = mu0(1);
mu02 = mu0(2);
mu11 = mu1(1);
mu12 = mu1(2);

a = invsame(1,1);
b = invsame(1,2);
c = invsame(2,1);
d = invsame(2,2);

a0=a;
b0=b;
c0=c;
d0=d;

a1=a;
b1=b;
c1=c;
d1=d;

phiterm = 2*log((1-phi)/phi); 
logterm = 0; %log(det(sigma)/det(sigma)); FOR LDA


%%plotting LDA %%
fn = sprintf('(p^2)*%f + (q^2)*%f - 2*p*%f - 2*q*%f - p*%f - q*%f + p*q*%f + %f',(a0-a1),(d0-d1),(mu01*a0 - mu11*a1),(mu02*d0 - mu12*d1),(mu02*(b0+c0) - mu12*(b1+c1)),(mu01*(b0+c0) - mu11*(b1+c1)),(b0+c0-b1-c1),((mu01^2)*a0 - (mu11^2)*a1) + ((mu02^2)*d0 - (mu12^2)*d1) + (mu01*mu02*(b0+c0) - mu11*mu12*(b1+c1)) - logterm - phiterm);
ez1 = ezplot(fn,[-3,3,-6,6]);
set(ez1,'color',[0 1 1]);
set(ez1,'LineWidth',2);

%%% QDA variables and PLOT %%%%%%%%%%%%%%%%%%%%%%%%%%%
a1 = inv1(1,1);
b1 = inv1(1,2);
c1 = inv1(2,1);
d1 = inv1(2,2);

a0 = inv0(1,1);
b0 = inv0(1,2);
c0 = inv0(2,1);
d0 = inv0(2,2);

logterm = log(det(sigma1)/det(sigma0));
disp('logterm');
disp(logterm);

%% why new func, because the variable has changed for a0, etc. %%%
%% plotting QDA %% 
fn2 = sprintf('(p^2)*%f + (q^2)*%f - 2*p*%f - 2*q*%f - p*%f - q*%f + p*q*%f + %f',(a0-a1),(d0-d1),(mu01*a0 - mu11*a1),(mu02*d0 - mu12*d1),(mu02*(b0+c0) - mu12*(b1+c1)),(mu01*(b0+c0) - mu11*(b1+c1)),(b0+c0-b1-c1),((mu01^2)*a0 - (mu11^2)*a1) + ((mu02^2)*d0 - (mu12^2)*d1) + (mu01*mu02*(b0+c0) - mu11*mu12*(b1+c1)) - logterm - phiterm);
ez2=ezplot(fn2,[-3,3,-6,6]);
set(ez2,'LineWidth',2);

%% setting aesthetic parameters %%%%%%%%%%%%%%%%%%%%%%
legend('Alaska','Canada','LDA','QDA');
xlabel('x1');
ylabel('x2');
title('Ques 4: LDA and QDA Binary Classifier');