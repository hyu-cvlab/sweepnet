function [out,dout] = rodrigues(in)
% function [R,dRdrot] = rodrigues(rot)
%   or     [rot,drotdR] = rodrigues(R)
%
% input:
%    rot(3)/R(3,3) : 3D rotation vector/matrix
%
% output:
%    R(3,3)/rot(3) : 3D rotation matrix/vector
%    dRdrot/drotdR : derivative of R/rot w.r.t. rot/R respectively
%
% from Jean-Yves Bouguet's code (originally from Pietro Perona 1993)
%

switch (length(in(:)))

case 3;  %% rotation vector
  rot = in(:);
  th = norm(rot);
  if (th < eps)
    out = eye(3);
    dout = [0,0,0; 0,0,1; 0,-1,0; 0,0,-1; 0,0,0; 1,0,0; 0,1,0; -1,0,0; 0,0,0];

  else
    om = rot/th;

    alpha = cos(th);
    beta = sin(th);
    gamma = 1-cos(th);

    M = [0, -om(3), om(2); om(3), 0, -om(1); -om(2), om(1), 0];
    A = om*om';
    out = eye(3)*alpha + M*beta + A*gamma;

    if (nargout > 1)
      dm3drot = [eye(3); om'];
      dm2dm3 = [eye(3)/th, -rot/th^2; zeros(1,3), 1];

      dm1dm2 = zeros(21,4);
      dm1dm2(1,4) = -sin(th);
      dm1dm2(2,4) = cos(th);
      dm1dm2(3,4) = sin(th);
      dm1dm2(4:12,1:3) = [0 0 0 0 0 1 0 -1 0;
                          0 0 -1 0 0 0 1 0 0;
                          0 1 0 -1 0 0 0 0 0]';
      dm1dm2(13:21,1) = [2*om(1); om(2); om(3); om(2); 0; 0; om(3); 0; 0];
      dm1dm2(13:21,2) = [0; om(1); 0; om(1); 2*om(2); om(3); 0; om(3); 0];
      dm1dm2(13:21,3) = [0; 0; om(1); 0; 0; om(2); om(1); om(2); 2*om(3)];

      dRdm1 = zeros(9,21);
      dRdm1([1 5 9],1) = ones(3,1);
      dRdm1(:,2) = M(:);
      dRdm1(:,4:12) = beta*eye(9);
      dRdm1(:,3) = A(:);
      dRdm1(:,13:21) = gamma*eye(9);

      dout = dRdm1 * dm1dm2 * dm2dm3 * dm3drot;
    end
  end

case 9;  %% rotation matrix
  R = reshape(in,[3,3]);
  [U,S,V] = svd(R);
  R = U*V';

  t = (trace(R)-1)/2;
  th = real(acos(t));

  if (sin(th) >= 1e-5)
    vth = 1/(2*sin(th));

    om1 = [R(3,2)-R(2,3), R(1,3)-R(3,1), R(2,1)-R(1,2)]';
    om = vth*om1;
    out = om*th;

    if (nargout > 1)
      dtdR = [1 0 0 0 1 0 0 0 1]/2;
      dthdt = -1/sqrt(1-t^2);
      dthdR = dthdt * dtdR;

      dvthdth = -vth*cos(th)/sin(th);
      dvar1dth = [dvthdth;1];
      dvar1dR =  dvar1dth * dthdR;

      dom1dR = [0 0 0 0 0 1 0 -1 0;
                0 0 -1 0 0 0 1 0 0;
                0 1 0 -1 0 0 0 0 0];
      dvardR = [dom1dR;dvar1dR];   %% var = [om1;vth;theta];
  
      % var2 = [om;theta];
      domdvar = [vth*eye(3), om1, zeros(3,1)];
      dthdvar = [0 0 0 0 1];
      dvar2dvar = [domdvar; dthdvar];

      domdvar2 = [th*eye(3), om];
      dout = domdvar2 * dvar2dvar * dvardR;
    end

  elseif (t > 0)   %% case norm(om)=0;
    out = [0 0 0]';

    dout = [0 0 0 0 0 1/2 0 -1/2 0;
            0 0 -1/2 0 0 0 1/2 0 0;
            0 1/2 0 -1/2 0 0 0 0 0];

  else             %% case norm(om)=pi; %% fixed April 6th
    out = th * (sqrt((diag(R)+1)/2).*[1;2*(R(1,2:3)>=0)'-1]);
    if nargout > 1,
      fprintf(1,'WARNING!!!! Jacobian drotdR undefined!!!\n');
      dout = NaN*ones(3,9);
    end
  end 

otherwise;
  error('invalid input rot/R');
end

