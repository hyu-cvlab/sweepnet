function [color_map] = makeColorMap(arr, m, M)

[h, w] = size(arr);
arr = double(arr);
samples = 100000;
if nargin < 3
    M = max(max(arr));
    if nargin < 2
        m = min(min(arr));
    end
end
colors = jet(samples);

arr = round((arr-m)/(M-m) * (samples-1)) + 1;
arr(isnan(arr)) = 1;
arr(arr<1) = 1;
arr(arr>samples) = samples;
R = reshape(colors(arr,1), [h w]);
G = reshape(colors(arr,2), [h w]);
B = reshape(colors(arr,3), [h w]);

color_map = cat(3, R, G, B);
color_map = uint8(round(255*color_map));
end

