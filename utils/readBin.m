function [x] = readBin(path, precision, shape, transpose)
%READBIN Summary of this function goes here
%   Detailed explanation goes here
if nargin < 4
    transpose = false;
end
nelem = cumprod(shape);
nelem = nelem(end);
if exist(path, 'file')
    f = fopen(path, 'rb');
    x = fread(f, nelem, precision);
    fclose(f);
else
   warning('# %s does not exist.', path);
   x = zeros(shape);
end
eval(['x = ' precision '(x);']);
x = reshape(x, shape);
if transpose
    x = x';
end
end

