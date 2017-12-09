
function [LLH] = model_llh(params,data,N,T);
p.rho1=params(1);
p.rho2=params(2);
p.phi1=params(3);
p.phi2=params(4);
p.beta=params(5);
p.sigmae=params(6);
p.sigmaA=params(7);
p.sigmaB=params(8);

T=min (T,length(data));
data_logA=log(data(:,1));
data_B=data(:,2);

rng(0)
lrsim=5000;
x_dist=zeros(lrsim+3,1);
distshock=p.sigmae*randn(lrsim+3,1);
for t=3:lrsim+3;
    x_dist(t)=p.rho1*x_dist(t-1)+p.rho2*x_dist(t-2)+p.phi1*distshock(t-1)+p.phi2*distshock(t-2);
end


particles=zeros(T,N,6);
llhs=zeros(T,1);
initial_sample=randsample(lrsim,N);

particles(1,:,1)=x_dist(initial_sample+2);
particles(1,:,2)=x_dist(initial_sample+1);
particles(1,:,3)=x_dist(initial_sample);
particles(1,:,4)=distshock(initial_sample+2);
particles(1,:,5)=distshock(initial_sample+1);
particles(1,:,6)=distshock(initial_sample);

llh=normpdf(data_logA(1),particles(1,:,1),p.sigmaA).* normpdf(data_B(1),p.beta*particles(1,:,1).^2,p.sigmaB);
llh(1)=log(sum(llh)-log(N));

for t=2:T
    shock=p.sigmae*randn(1,N);
    particles(t,:,1)=p.rho1*particles(t-1,:,1)+p.rho2*particles(t-1,:,2)+p.phi1*particles(t-1,:,4)+p.phi2*particles(t-1,:,5)+shock;
    particles(t,:,2)=particles(t-1,:,1);
    particles(t,:,3)=particles(t-1,:,2);
    particles(t,:,4)=shock;
    particles(t,:,5)=particles(t-1,:,4);
    particles(t,:,6)=particles(t-1,:,5);
    
    
    llh=normpdf(data_logA(t),particles(t,:,1),p.sigmaA) .* normpdf(data_B(t),p.beta*particles(t,:,1).^2,p.sigmaB);
    w=exp(log(llh)-log(sum(llh)));
    
    if sum(llh)==0
        w(:)=1/length(w);
    end
    
    llh_store(t)=log(sum(llh))-log(N);
    
    samples=randsample(N,N,true,w);
    particles(t,:,:)=particles(t,samples,:);
    
end

LLH=sum(llh_store)
end

