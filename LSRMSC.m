% Learning Smooth Representation for Multi-view Subspace Clustering
function [result, YY, ZV, FV, RV, mu] = LSRMSC(data, labels, eta, gamma, k, Iter,normData)
% multi-view data: cell array, view_num by 1, each array is num_samp by d_v
% labels: groundtruth of the data, num_samp by 1
% num_clus: number of clusters
% num_view: number of views
% num_samp: number of samples
% k: Order of the low-pass filter based on normalized Laplacian Fourier base
if nargin < 3
    eta = 1;
end
if nargin < 4
    gamma = 1;
end
if nargin < 5
    k = 2;
end
if nargin < 6
    Iter = 15;
end
if nargin < 7
    normData = 2;
end
num_view = size(data,1);
num_samp = size(labels,1);
num_clus = length(unique(labels));
mu = 1/num_view*ones(num_view,1);
opts.record = 0;
opts.mxitr  = 1000;%1000
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;
out.tau = 1e-3;
Aeq = ones(1,num_view);
Beq = 1;
lb = zeros(num_view,1);
%
% === Normalization1 ===
if normData == 1
    for i = 1:num_view
        dist = max(max(data{i})) - min(min(data{i}));
        m01 = (data{i} - min(min(data{i})))/dist;
        data{i} = 2 * m01 - 1;
    end
end
% === Normalization2 ===
if normData == 2
    for iter = 1:num_view
        for  j = 1:num_samp
            normItem = std(data{iter}(j,:));
            if (0 == normItem)
                normItem = eps;
            end
            data{iter}(j,:) = (data{iter}(j,:) - mean(data{iter}(j,:)))/normItem;
        end
    end
end
% === initialize === 
FV = cell(num_view, 1);
ZV = cell(num_view, 1);
RV = cell(num_view, 1); 
Rv = eye(num_clus);
for v = 1:num_view
    Zv = (data{v}*data{v}'+ eta*eye(num_samp))\(data{v}*data{v}');
    Zv(find(Zv<0)) = 0;
    Zv = (Zv + Zv')/2;
    Zv = Zv - diag(diag(Zv));
    Lv = diag(sum(Zv)) - Zv;
    [Fv, ~, ev] = eig1(Lv, num_clus, 0);
    FV{v} = Fv;
    ZV{v} = Zv;
    RV{v} = Rv;
end
% === iteration === 
fprintf('begin updating ......\n')
for iter = 1:Iter
    fprintf('the %d -th iteration ...... \n',iter) 
    %
    % === update Zv ===
    LV_norm{v} = cell(num_view, 1);
    data_bar = cell(num_view, 1);
    for v = 1:num_view
        Zv = ZV{v};
        Dv = diag(sum(Zv));
        Lv = eye(num_samp) - Dv^(-1/2) * Zv * Dv^(-1/2);
        LV_norm{v} = Lv;
        XV_bar = data{v};
        for i = 1:k % === k order ===
            XV_bar = (eye(num_samp) - Lv/2)*XV_bar;
        end
        temp = inv((XV_bar*XV_bar') + eta*eye(num_samp)); 
        data_bar{v} = XV_bar;
        XV = XV_bar';
        Fv = FV{v};
        for ij = 1:num_samp
            d = distance(Fv, num_samp, ij);
            XX = XV'*XV;
            Zv(:,ij) = temp*(XX(ij,:)' - (gamma/4)*d'); 
        end
        Zv(find(Zv<0)) = 0;
        Zv = (Zv + Zv')/2;
        Zv = Zv - diag(diag(Zv));
        ZV{v} = Zv;
    end
    %
    % === update Y ===
    if iter > 1
        Y_old = Y;
    end
    sumFR = zeros(num_samp, num_clus);
    for v = 1:num_view
        sumFR = sumFR + mu(v)*FV{v}*RV{v};
    end
    Y = zeros(num_samp, num_clus);
    [~, yy] = max(sumFR, [], 2);
    Y = full(sparse(1:num_samp, yy, ones(num_samp, 1), num_samp, num_clus));
    %
    % === update Rv ===
    SumFR = zeros(num_samp, num_clus);
    for v = 1:num_view
        SumFR = SumFR + mu(v)*FV{v}*RV{v};
    end
    for v = 1:num_view
        M = Y - SumFR + mu(v)*FV{v}*RV{v};
        [Wv, ~, Hv] = svd(mu(v)*FV{v}'*M);
        Rv = Wv*Hv';
        RV{v} = Rv;
    end    
    %
    % === update Fv === 
    SumFR = zeros(num_samp, num_clus);
    for v = 1:num_view
        SumFR = SumFR + mu(v)*FV{v}*RV{v};
    end
    LV = cell(num_view, 1);
    for v = 1:num_view
        Lv = diag(sum(ZV{v})) - ZV{v};
        LV{v} = Lv;
        M = Y - SumFR + mu(v)*FV{v}*RV{v};
        [FV{v}, out] = solveF(FV{v}, @fun1, opts, (1/gamma)/mu(v), M, mu(v)*RV{v}, LV{v});
    end
    % 
    % === update \mu === 
    Q = zeros(num_view, num_view);
    for ii = 1:num_view
        FRi = FV{ii}*RV{ii};
        for jj = 1:num_view
            FRj = FV{jj}*RV{jj};
            Q(ii,jj) = trace(FRi*FRj');
        end
    end
    for ii = 1:num_view 
        temp1 = (norm(data_bar{ii}'-data_bar{ii}'*ZV{ii},'fro'))^2 + eta*(norm(ZV{ii},'fro'))^2 + gamma*trace(FV{ii}'*LV{ii}*FV{ii});
        P(ii) = -1*temp1 + 2*trace(Y'*FV{ii}*RV{ii});
    end
    mu = fun_alm(Q, P); % paramu 
    %
    % objective value
    obj = 0;
    for ii = 1:num_view 
        obj = obj + mu(ii)*((norm(data_bar{ii}'-data_bar{ii}'*ZV{ii},'fro'))^2 + eta*(norm(ZV{ii},'fro'))^2 + gamma*trace(FV{ii}'*LV{ii}*FV{ii}));
    end
    SumFR = zeros(num_samp, num_clus);
    for ii = 1:num_view
        SumFR = SumFR + mu(ii)*FV{ii}*RV{ii};
    end
    obj = obj + (norm(Y - SumFR,'fro'))^2; 
    OBJ(iter) = obj;
    %
    mu(find(mu<=0)) = 0.01; 
   %
   if (iter > 3) && ((norm(Y-Y_old,'fro')/norm(Y_old,'fro')) < 1e-4)
       break
   end 
   
   for i = 1:num_clus
       Y(:,i) = i*Y(:,i);
   end
   YY = sum(Y,2); 
   result = Clustering8Measure(labels, YY);
end 
% result = [nmi ACC Purity Fscore Precision Recall AR Entropy];
   for i = 1:num_clus
       Y(:,i) = i*Y(:,i);
   end
   YY = sum(Y,2);
   result = Clustering8Measure(labels, YY);

end

function [all] = distance(F,n,ij)
  for ji = 1:n
      all(ji) = (norm(F(ij,:)-F(ji,:)))^2;
  end
end   

function [F,G] = fun1(P,alpha,Y,Q,L)
    G = 2*L*P - 2*alpha*Y*Q';
    F = trace(P'*L*P) + alpha*(norm(Y-P*Q,'fro'))^2;
end