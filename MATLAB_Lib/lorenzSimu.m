function [ u,t ] = lorenzSimu( t,alpha,beta,rho )
    % http://rin.io/matlab-lorenz-attractor/
    if nargin < 4
        alpha = 10; beta = 8/3; rho = 28;
    end
    lor2 = @(t,y)[alpha*(y(2)-y(1)); y(1)*(rho-y(3))-y(2); y(1)*y(2)-beta*y(3)];
    [t,u] = ode45(lor2,[0,t],100*(rand(3,1)-0.5));
    
    plot3(u(:,1),u(:,2),u(:,3),'DisplayName',sprintf(...
        '\\alpha=%.2f,\\beta=%.2f,\\rho=%.2f',alpha,beta,rho));
    box on; grid on; xlabel('X'); ylabel('Y'); zlabel('Z');
    legend show;
end

