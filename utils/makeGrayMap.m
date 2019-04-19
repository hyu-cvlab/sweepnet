function [I] = makeGrayMap(arr, m, M)

[h, w] = size(arr);
arr = double(arr);
if nargin < 3
    M = max(max(arr));
    if nargin < 2
        m = min(min(arr));
    end
end

arr = round((arr-m)/(M-m) * 255);
arr(isnan(arr)) = 1;
I = round(uint8(reshape(arr, [h w])));

end

