clc; close all; clear; more off;

y = load('digitlabels2.txt');
X = load('digitdata2.txt');
K=4;

minJ = Inf;
Jfinal = [];
minC = [];
minratiomatrix = [];
for i = 1:1
  disp('Random init ');
  disp(i);
  [J c rmatrix] = helper(X,y);
  if minJ > J(size(J,1),:)
    minJ = J(size(J,1),:);
    minC = c;
    Jfinal = J;
    minratiomatrix = rmatrix;
  endif
  disp(J(size(J,1),:));
endfor

##work on minJ and minC now
figure(1);
plot(Jfinal); hold on;
xlabel('Number of iterations');
ylabel('Distortion Cost');
title('Part c');
disp(minJ(size(minJ,1),:)); ##last value!
hold off;


figure(2);
plot(minratiomatrix);
hold on;
xlabel('Number of iterations');
ylabel('Ratio of misclassified against total');
title('Part d');
hold off;