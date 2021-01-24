% A = randn(3,1000);
% A = A * diag(1./sqrt(sum(A.^2, 1)));
%sphere(40);
%hold on
%subplot(1,2,1)
%subplot(1,2,1)
r = 1;
[A,X,Y,Z] = generate_saddle(r);
%plot3(A(1,:),A(2,:),A(3,:),'o');
%x = randn(3,1);

%x = [0.1,0.1,-0.2]';
x = [0,0,saddle(0,0)]';
%x = [1/3, 1/3, saddle(1/3,1/3)]';
%x = x/norm(x);
subplot(1,2,1)
run(x,A,X,Y,Z,r);
axis([ -1 1 -1.5 1.5 -0.5 0.5])
x = [1/3, 1/3, saddle(1/3,1/3)]';
subplot(1,2,2)
run(x,A,X,Y,Z,r)
axis([ -1 1 -1.5 1.5 -0.5 0.5])

function run(x,A,X,Y,Z,r)
    plot3(X(:),Y(:),Z(:),'*')
    hold on
    %scatter3(x(1),x(2),x(3));
    h = 0.1;
    [Co_h, Co_v, U_h, U_v] = coordinate(x, A, h);
    %hold on
    Y = U_h*Co_h+x;
    %scatter3(Y(1,:),Y(2,:),Y(3,:));
    centered = A-repmat(x,[1,size(A,2)]);
    [~,ind] = sort(sum(centered.^2,1));
    [Tensor,N] = Tensor_fitting(Co_v, Co_h, 2, ind, centered, h);
    hold on
    %new_Y = New_Co_v(squeeze(Tensor),Co_h);
    %n_Y = U_v*new_Y(:,ind(1:300))+U_h*Co_h(:,ind(1:300))+x;
    [X,Y,Z] = grid_plot(squeeze(Tensor),1);
    new_cor = [U_h,U_v]*[X(:)';Y(:)';Z(:)']+x;
    %subplot(1,2,2)
    hold on
    surf(reshape(new_cor(1,:),size(X)),reshape(new_cor(2,:),size(X)),reshape(new_cor(3,:),size(X)))
end
%plot3(n_Y(1,:),n_Y(2,:),n_Y(3,:),'*');
%subplot(1,2,2)
%plot(N,'*')

function [data,X,Y,Z] = generate_saddle(r)
    x = -r:0.1:r;
    y = -r:0.1:r;
    [X,Y] = meshgrid(x,y);
    noise = 0*randn(size(X));
    Z = 0.5*(X.^2/1-Y.^2/1)+noise;
    %mesh(X,Y,Z)
    data = [X(:)';Y(:)';Z(:)'];
    %hold on
    %plot3(X,Y,Z,'*')
end


function z = saddle(x,y)
    z = 0.5*(x^2/1-y^2/1);
end


function [Co_h, Co_v, U_h, U_v] = coordinate(x, A, h)
    C = zeros(size(x,1),size(x,1));
    for i = 1: size(A,2)
        a = (x-A(:,i));
        C = C + exp(-norm(a)^2/h)*a*a';
    end
    [V,L,~] = svd(C);
    U_h = V(:,1:2);
    U_v = V(:,3);
    Co_h = U_h'*(A-repmat(x,1,size(A,2)));
    Co_v = U_v'*(A-repmat(x,1,size(A,2)));
end


function [Tensor, N] = Tensor_fitting(Co_v, Co_h, Tangent_dim, ind, centered, h)
    N = [];
    t_dim = size(Co_v,1);
    Tensor = zeros(t_dim, Tangent_dim, Tangent_dim);
    for r = 1:t_dim
%         X = squeeze(Tensor(r,:,:));
%         %k = 0;
%         while norm(Gradient(Co_h, X, Co_v(r,:), ind, centered)) > 0.01 
%             N = [N, norm(Gradient(Co_h, X, Co_v(r,:), ind, centered))];
%             X = X - 0.001*Gradient(Co_h, X, Co_v(r,:), ind, centered);
%         %    k = k+1;
%         end
        W = build_W(centered,h);
        [~, Theta] = least_square(Co_h, Co_v(r,:)', W);
        Tensor(r,:,:) = Theta;
    end
end

function W = build_W(centered,h)
    W = zeros(size(centered,2));
    for k = 1:size(centered,2)
        W(k,k) = exp(-norm(centered(:,k))^2/h);
    end
end

function [theta, Theta] = least_square(Tau, y, W)
    d = size(Tau, 1);
    ind = triu(true(size(Tau, 1)));
    d2 = d*(d+1)/2;
    G = zeros(d2, size(Tau,2));
    for i = 1:size(Tau,2)
        A = Tau(:,i)*Tau(:,i)';
        G(:,i) = A(ind);
    end
    theta = (G*W*G')\G*W*y;
    
    Theta_temp = zeros(d,d);
    Theta_temp(ind) = theta/2;
    Theta = Theta_temp+Theta_temp';
end

function X_g = Gradient(Co_h, X, t, ind, centered)
        X_g = zeros(size(X,2));
        n_sample = size(Co_h, 2);
        %[~,ind] = sort(sum(Co_h.^2,1));
        for k = 1:n_sample
            theta_k = Co_h(:,ind(k));
            X_g = X_g +exp(-norm(centered(:,ind(k)))^2/0.3)*((theta_k'*X*theta_k)*(theta_k*theta_k')-t(ind(k))*theta_k*theta_k');
%            X_g = X_g + (theta_k'*X*theta_k)*(theta_k*theta_k')-t(ind(k))*theta_k*theta_k';
        end
end

function Y = New_Co_v(Tensor,Co_h)
    Y = zeros(1,size(Co_h,2));
    for i = 1:size(Co_h,2)
        Y(i) = Co_h(:,i)'*Tensor *Co_h(:,i);
    end
end


function [X,Y,Z] = grid_plot(T,r)
    x = -r:0.1:r;
    y = -r:0.1:r;
    [X,Y] = meshgrid(x,y);
    z = New_Co_v(T,[X(:)';Y(:)']);
    Z = reshape(z, size(X));
end