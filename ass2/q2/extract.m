clear; close all; clc;
load('mnist_all.mat')
dim = [size(train0);size(train1);size(train2);size(train3);size(train4);size(train5);size(train6);size(train7);size(train8);size(train9)];
train = [train0;train1;train2;train3;train4;train5;train6;train7;train8;train9];
%train = double(train)/255.0;
%disp(train);
n = dim(1,2) + 1;
m = sum(dim);
m = m(1);

label = zeros(m,1);
sum = 1;
for i=1:10
    for j=1:dim(i,1)
      label(sum,1) = i; 
      sum = sum + 1;
    end
end

save('mnist_bin.mat','label','train');

clear; close all; clc;
load('mnist_all.mat');

dim2 = [size(test0);size(test1);size(test2);size(test3);size(test4);size(test5);size(test6);size(test7);size(test8);size(test9)];
test_data = [test0;test1;test2;test3;test4;test5;test6;test7;test8;test9];
%test = double(test)/255.0;

m2 = sum(dim2);
m2 = m2(1);

test_label = zeros(m2,1);
sum = 1;
for i=1:10
    for j=1:dim2(i,1)
      test_label(sum,1) = i; 
      sum = sum + 1;
    end
end


save('test_bin.mat', 'test_label', 'test_data');