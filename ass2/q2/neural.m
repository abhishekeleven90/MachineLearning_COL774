%% init stuff
clear ; close all; clc; more off;

in_size  = 784;  % 28 by 28
hi_size = 100;  % hidden 

%num_output = 10;        % 1-10
num_output = 2;          % 1, 2   

%trainfile = 'mnist_bin.mat';
trainfile = 'mnist_bin38.mat';

%testfile = 'test_bin.mat';
testfile = 'test_bin38.mat';

%config parameters
orig_eta = 1;
numIter = 0;
epsilon = 0.0001;
estart = 0.11; %%to define the random weights
                        
%% load data
load(trainfile);

%% define X and y
X = double(train)/255; % 255 is necessary to be done
y = label;
[m n] = size(X);

%% initial theta values
hh = hi_size;
ii = in_size;
ll = num_output;

%%TODO remove the comments from here
initial1 = 2*rand(hh,1+ii)*estart-estart; %randomW(in_size, hi_size); 
initial2 = 2*rand(ll,1+hh)*estart-estart; %randomW(hi_size, num_output);

it1 = initial1;
it2 = initial2;


%%for randomizing the data set
perm = randperm(m);
X=X(perm,:);
y=y(perm,:);


disp('Stochastic based on iteration starts!');
change = 0;
lastj = 0;
for i=1:numIter
    eta = 1/sqrt(i)*orig_eta;
    [j2 it1 it2] = stochastic_helper(it1, it2, in_size, hi_size,num_output, X, y, eta);
    disp(sprintf('Iter:%d \t Cost:%d \t eta:%d change:%d',i,j2,eta,change));
    change = abs(j2-lastj);
    lastj=j2;
end
disp('Stochastic based on iteration ends');

% pause;
disp('Stochastic based on error starts');

change = 1; %%difference between last J and new J
i = 1; %%iteration variable just to see which iteration going on
lastj = 0; %%to store last J value
starttime = cputime;
while (change>epsilon) %%while converging
    eta = 1/sqrt(i)*orig_eta;
    %% the j here is now error cost
    [j2 it1 it2] = stochastic_helper(it1, it2, in_size, hi_size,num_output, X, y, eta);
    %%disp(sprintf('Iter:%d \t Cost:%d \t eta:%d change:%d',i,j2,eta,change));
    change = abs(j2-lastj);
    lastj=j2;
    i = i+1;
    disp(sprintf('Working...Please wait. Change: %d',change));
end
endtime = cputime;
disp(sprintf('Time taken to learn: %d seconds',(endtime-starttime)));
disp('Stochastic based on error stops!');

%% accuracy
disp('Calculating Accuracy Now!');
load(testfile);
pred = accuracy(double(test_data)/255, it1, it2); %%test_data is our x, our input
correct = (test_label==pred);
correct = double(correct);
meanval = mean(correct);
disp(sprintf('Accuracy against testing set: %f', meanval*100));
disp('THE END');
