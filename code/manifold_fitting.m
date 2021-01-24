%[Data,R,B] = generate_swissroll();%generate_circle();
[Data,R,B] = generate_circle();

x = [1;0];
x = x./norm(x); 

%% Tensor Fitting
% sigma = sqrt(0.4);
% k = 10;
% [Theta, Co_h, Co_v, U_h, U_v] = fitting(x, Data, sigma);


%% plot fitted curve
% nx = -1.6:0.01:1.6;
% new_ = U_v*nx.*Theta{1}.*nx+U_h*nx+x;
% hold on
% plot(new_(1,:),new_(2,:),'-');


%% generate noise samples
% points = triangle_curve(U_v, U_h, x);
% plot(points(1,:),points(2,:),'bd','MarkerFaceColor','b');



%% projection backwards onto fitted manifold

P1 = sqrt(0.01:0.02:0.81);
P2 = 4:1:20;
%[X,Y] = meshgrid(P1,P2);
points = B;
error = zeros(length(P1),length(P2));
for s = 1:length(P1)
    for j = 1:length(P2)
        sigma = P1(s);
        k = P2(j);
        [e,~, ~,~] = implement_once(sigma, k, Data, R, B, 0);
        error(s,j) = e; 
        %axis([-1.8 1.8 -1.8 1.8])
    end
end
[r,c] = find(error==min(error(:)));
[e2, e3, Proj1, Proj2] = implement_once(P1(r),P2(c), Data, R, B,1);
%e2 = implement_once(0.4, 11, Data, R, B,1);
fprintf('Best Neighbor Size %d, Best h:%f\n',P2(c),P1(r));
%% show and plot
subplot('position',[0.04 0.06 0.45 0.9])
title('Nonlinear Projection')
plot(Data(1,:),Data(2,:),'k*','MarkerSize',8);
% hold on
hold on
plot(B(1,:),B(2,:),'rd','MarkerSize',6,'MarkerFaceColor','r');
hold on
plot(R(1,:),R(2,:),'r-');
plot(Proj1(1,:),Proj1(2,:),'bo','MarkerFaceColor','b');

subplot('position',[0.53 0.06 0.45 0.9])
title('Linear Projection')
plot(Data(1,:),Data(2,:),'k*','MarkerSize',8);
% hold on
hold on
plot(B(1,:),B(2,:),'rd','MarkerSize',6,'MarkerFaceColor','r');
hold on
plot(R(1,:),R(2,:),'r-');
plot(Proj2(1,:),Proj2(2,:),'bo','MarkerFaceColor','b');


%%
figure
imagesc(P1,P2, error);
colorbar

%points*diag(1./sqrt(sum(points.^2,1)));
e_o = mean(sqrt(sum((R-B).^2,1)));
fprintf('Percentage improved after nonlinear fitting:%f: linear fitting%f\n',1-e2/e_o, 1-e3/e_o);


function [e, e_l, new_points, new_points_linear] = implement_once(sigma, k, Data, R, points, draw)
        
        new_points = zeros(size(points));
        new_points_linear = zeros(size(points));
        for i = 1:size(points,2)
            x = shift_mean(points(:,i), Data, sigma, k);
            [Theta, ~, ~, U_h, U_v] = fitting(x, Data, sigma);
            [~, x_new, ~] = nonlinear_solution(U_h, U_v, points(:,i), x, Theta);
            new_points(:,i) = x_new;
            new_points_linear(:,i) = x + U_h*U_h'*points(:,i);
        end
%         if draw == 1
%             plot(new_points(1,:),new_points(2,:),'bo','MarkerFaceColor','b');
%         end

        %new_points*diag(1./sqrt(sum(new_points.^2,1)));
        e = mean(sqrt(sum((R-new_points).^2,1)));
        e_l = mean(sqrt(sum((R-new_points_linear).^2,1)));
end

function [Theta, Co_h, Co_v, U_h, U_v] = fitting(x, Data, sigma)

    %mean = shift_mean(x, Data, sigma, k);
    [Co_h, Co_v, U_h, U_v] = coordinate(x, Data, sigma);
    W = build_W(x, Data, sigma);
    [~, Theta] = least_square(Co_h, Co_v, W);
end

function mean = shift_mean(x, Data, h, k)
    [~,ind] = sort(sum((Data-x).^2,1),'ascend');
    mean = zeros(size(x));
    s_weight = 0;
    for i = 1:k
        w = exp(-norm(Data(:,ind(i))-x)^2/(h^2));
        s_weight = s_weight+w;
        mean = mean+w*Data(:,ind(i));
    end
    mean = mean/s_weight;
end

function points = triangle_curve(U_v, U_h, x)
    nx = -3:0.2:3;
    %points = U_v*(nx.*Theta{1}.*nx-0.4)+U_h*nx+x;
    points = 0.3*U_v*(cos(2*nx)-1.3)+0.5*U_h*nx+x;
end

%plot(x_new(1),x_new(2),'r*');


function [A, R, B] = generate_circle()
    theta = -pi:0.1:pi;
    x = cos(theta);
    y = sin(theta);
    n = [cos(theta);sin(theta)];
    n = n*diag(1./sum(n.^2,1));
    A = [x;y]+0.05*n*diag(randn(1,length(theta)));
    
    theta = -pi:0.2:pi;
    x = cos(theta);
    y = sin(theta);
    R = [x;y];
    n = [cos(theta);sin(theta)];
    n = n*diag(1./sum(n.^2,1));
    B = [x;y]+0.2*n*diag(randn(1,length(theta)));
end

function [A,R,B] = generate_swissroll()
    theta = -2*pi:0.1:2*pi;
    x = theta.*cos(theta);
    y = theta;
    R = 0.2*[x;y];
    n = [ones(1,length(theta));theta.*sin(theta)-cos(theta)];
    n = n*diag(1./sum(n.^2,1));
    A = 0.2*[x;y]+0.05*n*diag(randn(1,length(theta)));
    B = 0.2*[x;y]+0.2*n*diag(randn(1,length(theta)));
end

function [Co_h, Co_v, U_h, U_v] = coordinate(x, A, h)
    C = zeros(size(x,1),size(x,1));
    for i = 1: size(A,2)
        a = (x-A(:,i));
        C = C + exp(-norm(a)^2/(h^2))*a*a';
    end
    [V,L,~] = svd(C);
    U_h = V(:,1);
    U_v = V(:,2);
    Co_h = U_h'*(A-repmat(x,1,size(A,2)));
    Co_v = U_v'*(A-repmat(x,1,size(A,2)));
end


function W = build_W(x, A, h)
    centered = A-repmat(x,[1,size(A,2)]);
    W = zeros(size(centered,2));
    for k = 1:size(centered,2)
        W(k,k) = exp(-norm(centered(:,k))^2/(h^2));
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
