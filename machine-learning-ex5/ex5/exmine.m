clear;

p = 4;

X = [1; 2; 3]
X_poly = zeros(numel(X), p);

for i = 1:4,
  i
  X_poly(:,i) = X .^ i
end
