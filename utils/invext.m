function [ ext] = invext( ext )
rot = ext(1:3);
tr = ext(4:6);

R = rodrigues(-rot);
T = -R*tr;

ext = [rodrigues(R); T];


end

