function [v,obj] = fun_alm(A, b, paramu)
if nargin < 3
    paramu = 30;
end

if size(b,1) == 1
    b = b';
end

% initialize
rho = 1.5;
n = size(A,1);
alpha = ones(n,1);
v = ones(n,1)/n;
% obj_old = v'*A*v-v'*b;

obj = [v'*A*v-v'*b];
iter = 0;
while iter < 10
    % update z
    z = v-A'*v/paramu+alpha/paramu;

    % update v
    c = A*z-b;
    d = alpha/paramu-z;
    mm = d+c/paramu;
    v = EProjSimplex_new(-mm);

    % update alpha and mu
    alpha = alpha+paramu*(v-z);
    paramu = rho*paramu;
    iter = iter+1;
    obj = [obj;v'*A*v-v'*b];
end
end

function [x] = EProjSimplex_new(v, k)  % ���������շ���KKT�����������
%
% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=k
%
if nargin < 2
    k = 1;   %Լ��2 ת�á�1=1
end

ft=1;
n = length(v);

v0 = v-mean(v) + k/n;   % ������ �����V��ת������u���������
%vmax = max(v0);
vmin = min(v0);   % ����V0 ����С��Ԫ��
if vmin < 0  %������������������������
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;    %  v��Ϊ���� ÿһ��Ԫ�ض���ȥlambda ��23�� matlab�б���lambda�Զ������ͬά����
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g; % ������Ѱ�ң�24�����̵�0��  ��ţ�ٷ�
        ft=ft+1;
        if ft > 100   % ��������
            x = max(v1,0);
            break;
        end
    end
    x = max(v1,0);   % ���ģ�23��

else
    x = v0;
end
end

