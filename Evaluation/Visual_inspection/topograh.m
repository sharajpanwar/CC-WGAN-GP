% to create topograph or topo plots, we need EEGLab
% One way to get it is to clone this github repo
% https://github.com/ZijingMao/baselineeegtest and run add_curr_path.m
% this will add the repo to current path and then you can type "eeglab" in
% matlab command line 

% load the data and permute it to (channel, timestep, no. samples) format 
load('dir/Target_gen')
% load the channel information file eloc.ced'
load('dir/eloc.ced')
tp = permute(Target_gen, [2,3,1]); % we have 64x64x1000 now 
% upload data in EEG LAB
% let's average the data for 20 to 26 time steps (after 300 ml - 400 ml approx) 
% this will give topography over 60 ml range of event onset
tp_event = mean(tp(:,20:26,:), 2);
% topography over all samples
topo_samples = mean(tp_event, 3);
% finally ploting the topo plot using EEGLab 
topoplot(topo_samples, 'eloc.ced');
