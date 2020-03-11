%Normalize X2 RSVP data and generate ERP
% Data format channel * timepoint * #_of_epochs

% now normalize target epochs
[channels,timepoints,num_epochs] = size(target_ep);
target_ep = reshape(target_ep,[channels,timepoints*num_epochs]);
[~,mu,sigma] = zscore(target_ep');
target = (normalize(double(target_ep)',mu, sigma))';
% reorganize data as number_of_epochs * 2 * (channels*timepoints)
%input = permute(input,[2,3,1]);
target = reshape(target, [channels, timepoints, num_epochs]);

% now normalize nontarget epochs
[channels,timepoints,num_epochs] = size(nontarget_ep);
nontarget_ep = reshape(nontarget_ep,[channels,timepoints*num_epochs]);
[~,mu,sigma] = zscore(nontarget_ep');
nontarget = (normalize(double(nontarget_ep)',mu, sigma))';
% reorganize data as number_of_epochs * 2 * (channels*timepoints)
%input = permute(input,[2,3,1]);
nontarget = reshape(nontarget, [channels, timepoints, num_epochs]);


mean_target_input = mean(target,3);
figure (5); plot(mean_target_input'); figure(6); imagesc(mean_target_input)
mean_nontarget_input = mean(nontarget,3);
figure (7); plot(mean_nontarget_input'); ylim([-0.25 0.25]); figure(8); imagesc(mean_nontarget_input)

