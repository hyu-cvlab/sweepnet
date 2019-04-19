function [rot,tr,dpdp1,dpdp2] = composemotion(rot1,tr1,rot2,tr2)
% function [rot,tr,dpdp1,dpdp2] = composemotion(rot1,tr1,rot2,tr2)
%   or     [rot,tr,dpdp1,dpdp2] = composemotion(ext1,ext2)
%
% output:
%    rot(3), tr(3) : composed motion
%    dpdp1 = [ drdr1, drdt1; dtdr1, dtdt1 ];
%    dpdp2 = [ drdr2, drdt2; dtdr2, dtdt2 ];
%

if (nargin == 2 && extparam(rot1) && extparam(tr1))
  [rot2,tr2] = extparam(tr1);  [rot1,tr1] = extparam(rot1);
elseif (nargin == 3 && extparam(rot1))
  tr2 = rot2;  rot2 = tr1;  [rot1,tr1] = extparam(rot1);
elseif (nargin == 3 && extparam(rot2))
  [rot2,tr2] = extparam(rot2);
elseif (nargin ~= 4)
  error('invalid input parameters');
end

[R1,dR1dr1] = rodrigues(rot1);
[R2,dR2dr2] = rodrigues(rot2);

R = R2 * R1;
[rot,drdR] = rodrigues(R);
tr = R2*tr1(:) + tr2(:);

if (nargout <= 1)
  rot = [rot(:); tr(:)];
end
if (nargout > 2)
  [dRdR2,dRdR1] = dAB(R2,R1);

  drdr1 = drdR * dRdR1 * dR1dr1;
  drdr2 = drdR * dRdR2 * dR2dr2;
  drdt1 = zeros(3,3);
  drdt2 = zeros(3,3);

  [dtdR2,dtdt1] = dAB(R2,tr1(:));
  dtdt2 = eye(3,3);
  dtdr2 = dtdR2 * dR2dr2;
  dtdr1 = zeros(3,3);

  dpdp1 = [ drdr1, drdt1; dtdr1, dtdt1 ];
  dpdp2 = [ drdr2, drdt2; dtdr2, dtdt2 ];
end



function [dABdA,dABdB] = dAB(A,B)
%      [dABdA,dABdB] = dAB(A,B);
%      returns : dABdA and dABdB

  [p,n] = size(A);
  [tmp,q] = size(B);
  if n ~= tmp,
    error(' A and B must have equal inner dimensions');
  end;

  if (issparse(A) ||  issparse(B) || p*q*p*n>625)
    dABdA = spalloc(p*q,p*n,p*q*n);
  else
    dABdA = zeros(p*q,p*n);
  end

  for i=1:q,
    for j=1:p,
    ij = j + (i-1)*p;
      for k=1:n,
        kj = j + (k-1)*p;
        dABdA(ij,kj) = B(k,i);
      end;
    end;
  end;

  if (issparse(A) ||  issparse(B) || p*q*n*q>625)
    dABdB = spalloc(p*q,n*q,p*q*n);
  else
    dABdB = zeros(p*q,q*n);
  end;

  for i=1:q
    dABdB((i*p-p+1:i*p)',(i*n-n+1:i*n)) = A;
  end;

