clear; clc; close all;
more off;
%%TODO : python

train_loc = 'data/train.data'; %%data should be correct nonad --> +1 ; ad ---> -1;
test_loc = 'data/test.data';

load(train_loc);
[m n] = size(train);
y = train(:,n);
X = train(:,1:(n-1));

%%same n used here!
load(test_loc);
testX = test(:,1:(n-1));
testY = test(:,n);

%%code for gaussian
model = svmtrain(y, X, '-s 0 -t 2 -c 1 -g 0.00025 -q'); %% -g 0.00025

%%code for linear    
%model = svmtrain(y, X, '-s 0 -t 0 -c 1 -q'); 

%prediction
[predict_label_L, accuracy_L, dec_values_L] = svmpredict(testY, testX, model);

%%save support vector indices
sv_indices = model.sv_indices;
save('sv_libsvm_indices','sv_indices'); %%diff between gaussian and linear accordingly.
