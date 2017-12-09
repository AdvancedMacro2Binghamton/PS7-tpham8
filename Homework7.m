load data;


N=1000;
T=400;

%%%%%%Priors%%%%%
prior.rho1=@(x) unifpdf(x,0,1);
prior.rho2=@(x) unifpdf(x,0,1);
prior.phi1=@(x) unifpdf(x,0,1);
prior.phi2=@(x) unifpdf(x,0,1);
prior.beta=@(x) unifpdf(x,4,7); %%%because you said beta should be big
prior.sigmae=@(x) lognpdf(x, -0.5 ,1);
prior.sigmaA=@(x) lognpdf(x, -0.5 ,1);
prior.sigmaB=@(x) lognpdf(x, -0.5, 1);
prior.all= @(p) log(prior.rho1(p(1)))+log(prior.rho2(p(2)))+log(prior.phi1(p(3)))+log(prior.phi2(p(4)))+...
    log(prior.beta(p(5)))+log(prior.sigmae(p(6)))+log(prior.sigmaA(p(7)))+log(prior.sigmaB(p(8)));

%%%%%%Proposals%%%%
prop.rho1= 0.05;
prop.rho2= 0.05;
prop.phi1= 0.05;
prop.phi2= 0.05;
prop.beta=0.05;
prop.sigmae= 0.05;
prop.sigmaA= 0.05;
prop.sigmaB= 0.05;
prop.all=[prop.rho1 prop.rho2 prop.phi1 prop.phi2 prop.beta prop.sigmae prop.sigmaA prop.sigmaB];

%%%%%%Initial Values for All parameters%%%%
initial_params=[0.1 0.1 0.3 0.2 6 1 0.8 1];

M=5000;
acc=zeros(M,1);

llh_store=zeros(M,1);
parameters(1,:)=initial_params;

log_prior=prior.all(parameters(1,:));
llh=model_llh(parameters(1,:), data , N , T);
llh_store(1)=log_prior + llh;

rng(0)
prop_chance=log(rand(M,1));
prop_step= randn(M,8);
for m=2:M
    prop_params= parameters(m-1,:) + prop_step(m,:).*prop.all;
    prop_prior=prior.all(prop_params);
    if prop_prior>-Inf
        prop_llh=model_llh(prop_params,data,N,T);
        llh_store(m)=prop_prior+prop_llh;
        if llh_store(m)-llh_store(m-1)>prop_chance(m)
            accept=1;
        else
            accept=0;
        end
    else
        accept=0;
    end
    
    if accept
        parameters(m,:)=prop_params;
        acc(m)=1;
    else
        parameters(m,:)=parameters(m-1,:);
        llh_store(m)=llh_store(m-1);
    end
    waitbar(m/M)
    
end

acceptance_rate= sum(acc)/length(acc);
label={'\rho_1','\rho_2','\phi_1','\phi_2','\beta','\sigma_E','\sigma_A','\sigma_B'}
figure
for i=1:8
    subplot(4,2,i)
    histogram(parameters(:,i),50)
    title(label{i});
end

        
    


