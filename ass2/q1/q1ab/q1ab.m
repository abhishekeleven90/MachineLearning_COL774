%load and init
clear; clc; close all;

train_loc = 'PARAG/train.data'; %%data should be correct nonad --> +1 ; ad ---> -1;
test_loc = 'PARAG/test.data';


load(train_loc);
train = train(randperm(size(train,1)),:);
[m n] = size(train);

y = train(:,n);
X = train(:,1:(n-1));

%%same n used here!
load(test_loc);
testX = test(:,1:(n-1));
testY = test(:,n);


[row,col] = size(X);
b = ones(1,row);
C = 1;

%%linear kernel
Kl = (X * X');

%%now matrix_q calcs can be done
matrix_q = (y * y') .* Kl; 


cvx_begin
            variable alphaLin(row)
            maximize((-1/2)*(alphaLin'*matrix_q*alphaLin) + b*alphaLin )
            subject to
                alphaLin'*y == 0
                0 <= alphaLin
                alphaLin <= C
cvx_end

%%part 1
sv = find(alphaLin>1e-4 & alphaLin<0.999);
numSv = size(sv,1);
disp('Number of support vectors');
disp(numSv);
save('LINEAR_CVX_SV','sv');

%%part 2
%now we need to find w* and b*

%%finding W
%%need to be computed before b
temp = (alphaLin.*y);
W =  X'*temp;

%%finding b
b_min = inf;
b_max = -inf;
for i = 1:row
    compare = W' * X(i,:)';
    if y(i,1)==1   
        if b_min > compare
            b_min = compare;
        end
    end
    if y(i,1)==-1   
        if b_max < compare
            b_max = compare;
        end
    end 
end
b_star = (-1/2)*(b_max + b_min);


%%now calc accuracy
predict = testY.*(testX*W+b_star);
acc = sum(predict>0)/size(testY,1);
disp(acc);
