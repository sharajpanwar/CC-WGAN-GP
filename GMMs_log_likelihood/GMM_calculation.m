% This script calculates the GMM likelihood of real samples, vanilla GAN generated samples,
% and CC-GAN generated samples for all 64 channels

% The procedure/code is same for Target and nonTarget samples. 
% Let's assume Target data

% Upload real target, vanilla GAN generated target, and CC-GAN generated target data. 

% In first, we will identify the number of fitted GMMs for all 64 channels of
% real target data and store them in gmm_count

gmm_count = zeros(1,64); %create an array to store 
for ch = 1:64;
    X= squeeze(T_ch64_s1_r2(:,ch,:));
% determining the no of GMMs using Bayes Information Criterion (BIC) method 
options = statset('Display','final');
GMModel = fitgmdist(X,2,'CovarianceType','diagonal','Options',options, 'RegularizationValue',0.1);

% Examine the BIC over varying numbers of components k, here we check for maximum 10 GMM component.
BIC = zeros(1,10);
GMModels = cell(1,10);
options = statset('MaxIter',500);
for k = 1:10;
    GMModels{k} = fitgmdist(X,k,'Options',options,'CovarianceType','diagonal','RegularizationValue',0.1);
    BIC(k)= GMModels{k}.BIC;
end

[minBIC,numComponents] = min(BIC);
numComponents ;% out put the number of GMM component fitted 
gmm_count(ch) = numComponents;
end

% Create an array of 64x4 to store no of fitted GMM, gmm likelihood of samples from real,
% Vanilla GAN, and CC_GAN 0f 64 channels

GMM_likelihood = zeros(64,4);
% we will use "GMM_likelihood_channel" function to calculate The GMM
% likelihood distance for each channel iteratively and store in
% "GMM_likelihood" array, please refer to the function script
% GMM_likelihood_channel.m

for n_ch = 1:64;
    GMM_likelihood_ch = GMM_likelihood_channel(T_ch64_s1_r2, target_gen, target_gen, n_ch, gmm_count);
    GMM_likelihood(n_ch,:) = GMM_likelihood_ch;
end

% Now you can have GMM_likelihood array 
