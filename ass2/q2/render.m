clc; close all; clear; more off;

sq = 28; %%can be generalized thus
gray_image = zeros(sq,sq); %%resultant image
vectorindex = 1;

%% Load the dataset
load('mnist_all.mat');

%% Get the file name as input
%% This can be train0, train1, test5, etc.
filename = train6;

%% A valid index based on the number of examples in data sets
index = 35;

%%convert that one row to a 28 by 28 matrix thats it
for i = 1:sq
  for j = 1:sq

      gray_image(i,j) = filename(index,vectorindex);
      vectorindex = vectorindex + 1;
      
  end
end

%%now show the image!
imshow(gray_image);