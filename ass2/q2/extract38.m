%% for binary classification (3,8)
%% for b part of q2 in assignment 2

%%init work
clc; clear; close all;

%%load data
load('mnist_all.mat');

%%get dimensions
[m1 n1] = size(train3);
[m2 n2] = size(train8);

%%set data to be saved
train = [train3;train8]; %make it double when in memory, else takes time
label = [ones(m1,1);ones(m2,1)+1]; % zero  3, one 8

%%save data
save('mnist_bin38.mat', 'label', 'train');

%%repeat all for testing data
[m1 n1] = size(test3);
[m2 n2] = size(test8);
test_data = [test3;test8];
test_label = [ones(m1,1);ones(m2,1)+1];
save('test_bin38.mat', 'test_label', 'test_data');