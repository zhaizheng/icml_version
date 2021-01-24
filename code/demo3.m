
Data = generate_data();
plot(Data(1,:),Data(2,:),'*','MarkerSize',8);
hold on
Orig = [1,0,0,-1;0,1,-1,0];
% x = [1;0];
% x = x./norm(x); 

% %% Tensor Fitting
 sigma = 0.1;
 k = 10;
% [Theta, Co_h, Co_v, U_h, U_v] = fitting(x, Data, sigma, k);
% 
% 
% %% plot fitted curve
% nx = -1.6:0.01:1.6;
% new_ = U_v*nx.*Theta{1}.*nx+U_h*nx+x;
% hold on
% plot(new_(1,:),new_(2,:),'-');
% 
% 
% %% generate noise samples
% points = triangle_curve(U_v, U_h, x);
% plot(points(1,:),points(2,:),'bd','MarkerFaceColor','b');
% 
% hold on

%% projection backwards onto fitted manifold

%points = Data;
for k = 1
    x = Orig(:,k);
    [Theta, Co_h, Co_v, U_h, U_v] = fitting(x, Data, sigma, k);
    points = triangle_curve(U_v, U_h, x);
    hold on
    plot(points(1,:),points(2,:),'bd','MarkerFaceColor','b');
    
    nx = -1.2:0.1:1.2;
    new_ = U_v*nx.*Theta{1}.*nx+U_h*nx+x;
    hold on
    plot(new_(1,:),new_(2,:),'-','Linewidth',3);
    
    new_points = zeros(size(points));

    for i = 1:size(points,2)
        %x = shift_mean(points(:,i), Data, sigma, k);
        [Theta, Co_h, Co_v, U_h, U_v] = fitting(x, Data, sigma, k);
        [~, x_new, error] = nonlinear_solution(U_h, U_v, points(:,i), x, Theta);
        new_points(:,i) = x_new;
    end
    plot(new_points(1,:),new_points(2,:),'ro','MarkerFaceColor','r');

    axis([-1.8 1.8 -1.8 1.8])
    hold on
end

%% project to the true manifold
% points_true = points*diag(1./sqrt(sum(points.^2,1)));
% plot(points_true(1,:),points_true(2,:),'go','MarkerFaceColor','g');

function [Theta, Co_h, Co_v, U_h, U_v] = fitting(x, Data, sigma, k)

    %mean = shift_mean(x, Data, sigma, k);
    [Co_h, Co_v, U_h, U_v] = coordinate(x, Data, sigma);
    W = build_W(x, Data, sigma);
    [~, Theta] = least_square(Co_h, Co_v, W);
end

function mean = shift_mean(x, Data, sigma, k)
    [~,ind] = sort(sum((Data-x).^2,1),'ascend');
    mean = zeros(size(x));
    s_weight = 0;
    for i = 1:k
        w = exp(-norm(Data(:,ind(i)))^2/sigma);
        s_weight = s_weight+w;
        mean = mean+w*Data(:,ind(i));
    end
    mean = mean/s_weight;
end

function points = triangle_curve(U_v, U_h, x)
    nx = -1.8:0.1:1.8;
    %points = U_v*(nx.*Theta{1}.*nx-0.4)+U_h*nx+x;
    points = 0.2*U_v*(cos(3*nx)-2)+0.5*U_h*nx+x;
end

%plot(x_new(1),x_new(2),'r*');


function A = generate_data()
    A = randn(2,150);
    A = A * diag(1./sqrt(sum(A.^2, 1)));
    
    A = A + 0.04*randn(2,150);
end


function [Co_h, Co_v, U_h, U_v] = coordinate(x, A, h)
    C = zeros(size(x,1),size(x,1));
    for i = 1: size(A,2)
        a = (x-A(:,i));
        C = C + exp(-norm(a)^2/h)*a*a';
    end
    [V,L,~] = svd(C);
    if V(:,1)'*x>0
        U_h = V(:,1);
    else
        U_h = -V(:,1);
    end
    if V(:,2)'*x>0
        U_v = V(:,2);
    else
        U_v = -V(:,2);
    end    
%     U_v = V(:,2);
    Co_h = U_h'*(A-repmat(x,1,size(A,2)));
    Co_v = U_v'*(A-repmat(x,1,size(A,2)));
end


function W = build_W(x, A, sigma)
    centered = A-repmat(x,[1,size(A,2)]);
    W = zeros(size(centered,2));
    for k = 1:size(centered,2)
        W(k,k) = exp(-norm(centered(:,k))^2/sigma);
    end
end

function [theta, Theta] = least_square(Tau, Co_v, W)
    d = size(Tau, 1);
    ind = triu(true(size(Tau, 1)));
    d2 = d*(d+1)/2;
    Theta = cell(1,size(Co_v,1));
    for j = 1:size(Co_v, 1)
        G = zeros(d2, size(Tau,2));
        for i = 1:size(Tau,2)
            A = Tau(:,i)*Tau(:,i)';
            G(:,i) = A(ind);
        end
        theta = (G*W*G')\G*W*Co_v(j,:)';

        Theta_temp = zeros(d,d);
        Theta_temp(ind) = theta/2;
        Theta{j} = Theta_temp+Theta_temp';
    end
end

function [tau,x_new, error] = nonlinear_solution(U_h, U_v, x, x0, Theta)
    s = U_h'*(x-x0);
    c = U_v'*(x-x0);
    d = size(U_h, 2);
    D = size(x,1);
    A = zeros(d,D-d);
    tau = s;
    tau_old = zeros(size(s));
    error = [];
    while norm(tau-tau_old)> 1.0e-14
        tau_old = tau;
        for i = 1:D-d
            A(:,i) = Theta{i}*tau;
        end
        tau = (A*A'+eye(d)/2)\(s/2+A*c);
        for i = 1:D-d
            A(:,i) = Theta{i}*tau;
        end
        tau = (A*A'+eye(d)/2)\(s/2+A*c);
        error = [error,norm(tau-tau_old)];
    end
    iota = zeros(D-d,1);
    for j = 1:D-d
        iota(j) = tau'*Theta{j}*tau;
    end
    x_new = U_h*tau+U_v*iota+x0;
end

function [tau1,tau2,x_new, error] = nonlinear_solution_bk(U_h, U_v, x, x0, Theta)
    s = U_h'*(x-x0);
    c = U_v'*(x-x0);
    d = size(U_h, 2);
    D = size(x,1);
    A = zeros(d,D-d);
    tau2 = s;
    tau1 = zeros(size(s));
    error = [];
    while norm(tau1-tau2)> 1.0e-8
        %tau_old = tau;
        for i = 1:D-d
            A(:,i) = Theta{i}*tau2;
        end
        tau1 = (A*A'+eye(d))\(s+A*c);
        for i = 1:D-d
            A(:,i) = Theta{i}*tau1;
        end
        tau2 = (A*A')\(A*c);
        error = [error,norm(tau1-tau2)];
    end
    iota = zeros(D-d,1);
    for j = 1:D-d
        iota(j) = tau2'*Theta{j}*tau2;
    end
    x_new = U_h*tau2+U_v*iota+x0;
end
