function error = L2error( y, yTest )
  error= sqrt(sum( (y - yTest).^2, "all"));
end
