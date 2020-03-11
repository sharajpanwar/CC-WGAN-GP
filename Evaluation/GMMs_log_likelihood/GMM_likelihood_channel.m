
% This function calculate the GMM likelihood of real samples, vanilla GAN generated samples,
% and CC-GAN generated samples of any given channel

% The GMMs are fitted on real data and average likelihood of real, vanilla
% GAN and CC-GAN is calculated to compare the sample's quality.
 
function GMM_likelihood_channel = Distance(real, Vanilla_GAN, CC_GAN, channel, gmm_count);
% real => real samples, Vanilla_GAN => Samples generated from vanilla GAN, CC_GAN => samples
% generated from CC-GAN, channel => channel to be evaluated, i.e. 30 for POz, 
% gmm_count => no of fitted GMM using BCI (please refer to GMM_calculation.m) 
real_channel= squeeze(real(:,channel,:));
Vanila_GAN_channel= squeeze(Vanilla_GAN(:,channel,:));
CC_GAN_channel= squeeze(CC_GAN(:,channel,:));
numComponents = gmm_count(channel);

gm =  fitgmdist(real_channel,numComponents,'RegularizationValue',0.1);
mu = gm.mu; sigma=gm.Sigma;
p = gm.ComponentProportion;

for i = 1:numComponents;
    d_real(:,i)=mvnpdf(real_channel, mu(i,:),sigma(:,:,i));
    d_Vanilla_GAN(:,i)=mvnpdf(Vanila_GAN_channel, mu(i,:),sigma(:,:,i));
    d_CC_GAN(:,i)=mvnpdf(CC_GAN_channel, mu(i,:),sigma(:,:,i));

end
% weighing wih the component proportion 
distance_Real = d_real*p';distance_Vanila_GAN = d_Vanilla_GAN*p';distance_CC_GAN = d_CC_GAN*p';
GMM_likelihood_Real = log(mean(distance_Real, 1));
GMM_likelihood_Vanilla_GAN = log(mean(distance_Vanila_GAN, 1));
GMM_likelihood_CC_GAN = log(mean(distance_CC_GAN, 1));

GMM_likelihood_channel = [numComponents, GMM_likelihood_Real, GMM_likelihood_Vanilla_GAN, GMM_likelihood_CC_GAN];
end
% function outputs a 1x4 array with no of fitted GMM, gmm likelihood of samples from real,
% Vanilla GAN, and CC_GAN 
