
% This script is to get the channel names and channelwise Target and nonTarget data 
% Loading the data files 
load('dir/Target');
load('dir/nonTarget');
load('dir/chanlocs.mat');

% creating a table of 64 channel names, this is useful for writeup too
 
channels = struct2cell(chanlocs64);
channels_names = channels(1,1,:)
channels_names=squeeze(channels_names);
channel_names_table =  cell2table(channels_names)

% Now creating a cell of channel wise Target (row1) and nonTarget (row2)

channel_data  = cell(1,64);

 
for i=1:64  
  channel_data{1,i}  = Target(:,i,:);
  channel_data{2,i}  = nonTarget(:,i,:);   
end
 
% these channels can be used to fit channel wise Support Vector Machine and
% Accuracy can be used as a measure to order them in terms of discriminative features
