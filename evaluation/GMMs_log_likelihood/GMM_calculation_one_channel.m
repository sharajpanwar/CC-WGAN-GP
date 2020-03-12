% Calculation for GMM Log-likelihood of real samples (Real~ GMM) vs 
% GMM Log-likelihood of generated samples(Gen~GMM) 
% let's calculate for target samples, same can be done for nonTarget

% Selecting a channel(ch) for GMM evaluation out of 64, i.e. POz

ch = 30;
X= squeeze(target_real(:,ch,:));

%determining the no of GMMs using BCI method

options = statset('Display','final');
GMModel = fitgmdist(X,2,'CovarianceType','diagonal','Options',options);

% Examine the BIC over varying numbers of components k, here we check for maximum 10 GMM component.

BIC = zeros(1,10);
GMModels = cell(1,10);
options = statset('MaxIter',500);
for k = 1:10
    GMModels{k} = fitgmdist(X,k,'Options',options,'CovarianceType','diagonal','RegularizationValue',0.1);
    BIC(k)= GMModels{k}.BIC;
end

[minBIC,numComponents] = min(BIC);
numComponents % out put the number of GMM component fitted 
% For examples, let's assume numComponents = 4

% GMM likelihood for 4 components
gm = fitgmdist(X,numComponents,'RegularizationValue',0.1);

X1=X; % likelihood distance from real samples
% X1 = squeeze(target_gen(:,ch,:)); uncomment for likelihhod distance from generated samples 

mu = gm.mu;
mu1=mu(1,:);mu2=mu(2,:);mu3=mu(3,:);mu4=mu(4,:);

sigma=gm.Sigma;
sigma1= sigma(:,:,1);sigma2= sigma(:,:,2);sigma3= sigma(:,:,3);sigma4= sigma(:,:,4);
% likelihood from GMM components
d1 = mvnpdf(X1,mu1,sigma1);d2 = mvnpdf(X1,mu2,sigma2);d3 = mvnpdf(X1,mu3,sigma3);d4 = mvnpdf(X1,mu4,sigma4);
d = [d1,d2,d3,d4];   
p = gm.ComponentProportion;
% weighing wih the component proportion 
distance = d*p';
Average_distance_real = log(mean(distance, 1))

% same procedure can be applied for other no of GMMs, all other channels, and nontarget samples 
