function [Y,dYdext] = rigidmotion(X,rot,tr);
% function [Y,dYdext] = rigidmotion(X,rot,tr);
%   or     [Y,dYdext] = rigidmotion(X,ext);
%
% input:
%    X(3,npts) : 3D point coordinates
%    rot(3),tr(3) : rigid motion parameter
%
% output:
%    Y(3,npts) : transformed 3D point coordinates
%    dYdext(3*npts,[3,3]) : derivative of Y w.r.t. to rot and tr
%
% Y = R * X + tr,  where R = rodrigues(rot)
%

if (nargin == 2 & extparam(rot))
  [rot,tr] = extparam(rot);
elseif (nargin ~= 3)
  error('invalid number of inputs');
end

if (size(X,1) ~= 3)
  error('invalid X');
end

npts = size(X,2);

if (nargout == 1)
  Y = rodrigues(rot) * X + repmat(tr(:),[1,npts]);

else
  [R,dRdrot] = rodrigues(rot);
  Y = R * X + repmat(tr(:),[1,npts]);

  dYdR = zeros(3*npts,9);
  dYdR(1:3:end, 1:3:end) = X';
  dYdR(2:3:end, 2:3:end) = X';
  dYdR(3:3:end, 3:3:end) = X';

  dYdext = zeros(3*npts,6);
  dYdext(:,1:3) = dYdR * dRdrot;
  dYdext(1:3:end,4) = 1;
  dYdext(2:3:end,5) = 1;
  dYdext(3:3:end,6) = 1;
end

