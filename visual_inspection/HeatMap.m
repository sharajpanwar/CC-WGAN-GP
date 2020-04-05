
% Creating Heat MAPS
% load the data in (no. samples, channel, timestep) format 
load('dir/Target_gen')
% average the samples and create 2D array of (channel, timestep => 64 x 64)
Target_avg= squeeze(mean(target_gen(:, :, :), 1));
% plot(HeatMap(Target_avg),'color',[0 0.4470 0.7410],'LineWidth',2, 'res', 300);
figure
M3= squeeze(mean(T_ch64_s1_r2(10, :, :), 1));
H_alerts3 = HeatMap(M3)
hFig3 = plot(H_alerts3);
title('Heat_map','FontSize', 24)
xlabel('Time Steps','FontSize', 24)
ylabel('Potential','FontSize', 24)
saveas(hFig3,'Heat_map','png'); 

