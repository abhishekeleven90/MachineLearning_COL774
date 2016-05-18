function ratio = getratio(cmatrix,y,K)
  toti = 0;
  tot = 0;
  for i = 1:K
      clustersubset = y(find(cmatrix==i),:);
      clusterclass = mode(clustersubset);
      correct = size(find(clustersubset==clusterclass),1);
      incorrect = size(clustersubset,1)-correct;
      toti = toti + incorrect;
      tot = tot + size(clustersubset,1);
  endfor
  ratio = toti/tot;
endfunction