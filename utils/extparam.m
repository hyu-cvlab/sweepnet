function [ret,tr] = extparam(param, tr, retvector)
% function ret = extparam(param)
%   or     param = extparam(param,retvector)
%   or     param = extparam(rot,tr,retvector)
%   or     [rot,tr] = extparam(param)
%

if (~isstruct(param) && length(param(:)) == 6)
  type = 1;  if (nargin>=2), retvector=tr; end;
elseif (size(param,1) == 6)
  type = 2;  if (nargin>=2), retvector=tr; end;
elseif (isstruct(param) && isfield(param,'rot') && isfield(param,'tr') && ...
    length(param(1).rot(:))==3 && length(param(1).tr(:))==3)
  type = 3;  if (nargin>=2), retvector=tr; end;
elseif (nargin > 1 && ((length(param(:))==3 && length(tr(:))==3) || ...
    (size(param,1)==3 && size(tr,1)==3 && size(param,2)==size(tr,2))))
  type = 4;
else
  type = 0;
end;

if (nargin > 1 || nargout > 1)  %% convert parameters
  switch (type)
  case 0;  error('invalid input arguments');
  case 1;  param = param(:);  rot = param(1:3);  tr = param(4:6);
  case 2;  rot = param(1:3,:);  tr = param(4:6,:);
  case 3;  rot = reshape([param(:).rot], [3,length(param)]);
           tr = reshape([param(:).tr], [3,length(param)]);
  case 4;  rot = param;
  end
  if (nargout > 1)
    if (exist('retvector','var') && ~retvector(1)), waring('conflict option'); end;
    ret = rot;  %tr = tr;
  elseif (exist('retvector','var') && retvector(1))
    ret = [rot; tr];
  else
    for i = 1:size(rot,2)
      ret(i).rot = rot(:,i)';  ret(i).tr = tr(:,i)';
    end
  end
else
  ret = (type > 0);
end

