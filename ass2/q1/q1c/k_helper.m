function kx1x2 = k_helper(x1,x2,gamma)
  xdiff = x1-x2 ;
  normval = xdiff*xdiff';
  kx1x2 = exp(-normval*gamma);
end

