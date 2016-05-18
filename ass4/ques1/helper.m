function [J c ratiomatrix] = helper(X,y)
  [m n] = size(X);

  K = 4;
  numiter = 20;
  Jmatrix  = zeros(m,1);
  J = []; ##sum of entries in Jmatrix
  ratiomatrix = [];
  converge_check = false; %%set this to true when you do not want to go to total of 30 iteratiuons!

  ##STEP 1  
  ##We have to choose K = 4 random rows
  mu = X(randperm(size(X, 1), K), :);
  oldmu = mu;
  
  ##STEP 2 
  ##Looping 30 times for now
  for i = 1:numiter
      oldmu = mu;

      #disp('Iteration number ');
      #disp(i,);
      
      ## STEP 2.1
      ## CENTROID ASSIGNMENT
      cmatrix = zeros(m,1);
      ##for each example
      for h = 1:m
          curr  = X(h,:);
          mindistance = Inf;
          mink = -1;
          ## for each centroid
          for k  = 1:K
              currk = mu(k,:);
              
              ##calc distance
              distance = norm(curr-currk)^2;
              if distance < mindistance
                  mindistance = distance;
                  mink = k;
              endif
            
          endfor
          cmatrix(h) = mink;
      endfor
      
      
      
      ##Calculating J here
      ##Distortion cost function
      diffmatrix = X-mu(cmatrix,:);
      for h =1:m
        Jmatrix(h,:) = norm(diffmatrix(h,:))^2;
      endfor
      
      #disp('Jmatrix size');
      #disp(size(Jmatrix));
      J = [J;sum(Jmatrix)];
      
      
      ##Calc ratio here
      ratiomatrix = [ratiomatrix;getratio(cmatrix,y,K)];
      
      
      ## STEP 2.2
      ## CENTROID MOVEMENT
      
      ##for each cluster
      for k = 1:K
        mu(k,:) = mean(X(find(cmatrix==k),:)); 
      endfor
      
      
      ##check for convergence here!
      if isequal(mu,oldmu) && converge_check==true
        disp('Converged at:');
        disp(i);
        break
      endif
    
  endfor
  
 

  c=cmatrix;
endfunction