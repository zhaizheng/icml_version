




function [theta, Theta] = least_square(Tau, y)
    d = size(Tau, 1);
    ind = triu(true(size(Tau, 1)));
    d2 = d*(d+1)/2;
    G = zeros(d2, size(Tau,2));
    for i = 1:size(Tau,2)
        A = Tau(:,i)*Tau(:,i)';
        G(:,i) = A(ind);
    end
    theta = (G*G')\G*y;
    
    Theta_temp = zeros(d,d);
    Theta_temp(ind) = theta/2;
    Theta = Theta_temp+Theta_temp';
end