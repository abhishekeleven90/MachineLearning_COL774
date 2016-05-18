%%init 
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

gamma = 0.00025; %%only to be used with the guassian method, K changes!
[row,col] = size(X);
b = ones(1,row);
C = 1;

%%kernel matrix
Kg = zeros(row,row);
%%for all training examples  with one another 
    
for i = 1:row 
  Kg(i,i) = 1;
    for j = i+1 : row
      %% xx and zz as in the notes
      xx = X(i,:);
      zz = X(j,:);
      
      %diff
      xdiff = xx-zz;
      %square
      norml = xdiff*xdiff';
      %exp
      Kg(i,j) = exp(-norml*gamma);
      %symmetric
      %i against j and j against i
      Kg(j,i) = Kg(i,j);
    end
end

%%now matrix_q calcs can be done
matrix_q = (y * y') .* Kg; 


cvx_begin
            variable alphaG(row)
            maximize((-1/2)*(alphaG'*matrix_q*alphaG) + b*alphaG )
            subject to
                alphaG'*y == 0
                0 <= alphaG
                alphaG <= C
cvx_end

%%part 1
sv = find(alphaG>1e-4 & alphaG<0.999);
numSv = size(sv,1);
disp('Number of support vectors');
disp(numSv);
save('GAUSSIAN_CVX_SV','sv');

%%part 2
%now we need to find w* and b*
%%in gaussian this part is very different than the linear one!

%%you cannot calculate W separately, instead you will follow eq 13 in notes
%%lets first find the alpha index that is maximum
maxAlpha = max(alphaG(sv));
maxIndex = find(alphaG==maxAlpha); %%index of max alpha
%%now we use this to calculate b_opt
xMax = X(maxIndex,:);
yMax = y(maxIndex,:);

%%sum_b = 0;
%%for i = 1:row
%%    sum_b = sum_b +  y(i,1) * alphaG(i,1) * GausianO(xMax,X(i,:),gamma); 
%%end
%%b_opt = yMax - sum_b; %%should come out to be 0.8377
%%disp('b_opt calculated');
%%disp(b_opt);

%%trying a diff way to constuct the K matrix
K_b_opt = zeros(row,1);
for i = 1:row
    K_b_opt(i,1) = k_helper(xMax,X(i,:),gamma);
end
b_opt = yMax - sum(y .* alphaG .* K_b_opt);
disp('b_opt calculated');
disp(b_opt);

%%now we need to predict and check against the testX values
%%let's first construct the output y matrix for the rwos
[trow tcol] = size(testX);
finalop = zeros(trow,1);

for i = 1:trow
   xCurr = testX(i,:);
   %%we need to fill the finalop(i,:)
   K_x_xi = zeros(row,1);
   for j = 1:row
       %%TODO: python script for segregating output
       %%TODO: Gaussian) correct!
        K_x_xi(j,1) = k_helper(xCurr,X(j,:),gamma);
   end
  finalop(i,:) = sum(y .* alphaG .* K_x_xi) + b_opt;
end

%%disp('used k_helper');
   


%%now calc accuracy
predict = testY.*finalop;
acc = sum(predict>0)/size(testY,1);
disp(acc);