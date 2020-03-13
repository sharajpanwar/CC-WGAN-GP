

% Creating sinusoidal by adding moderate noise for sample diversity and 
% moderate complexity 
sine_f5 = zeros(5000, 1, 64);
f=5;
Amp=1;
ts=1/63;
T=1;
t=0:ts:T;
noise = 0.15;

for n = 1:5000
    y=Amp*sin(2*pi*f*t) + noise*randn(1,64) ;
    sine_f5(n,:,:)=y;
end

% ploting the average data
y = squeeze(mean(sine_f5, 1));
plot(y)
% saving the data
save sine_f5.mat sine_f5 -v7.3

% To perform a wide range of experiments, the amplutide(A), frequency(f), and
% noise can be varied. A higher noise and frequncy will add more complexity 
% which will help to understand GAN behaviour on increasingly complex data






