clear all;
bins=[-20:0.1:20];
alpha = 1.5;
gam = 1;
n=10;
N=1E6;
fun = @(x,z) exp(-gam*x.^(alpha)).*cos(x*z);
for i=1:numel(bins)
  L(i)=1/pi*integral(@(x)fun(x,bins(i)),0,Inf);
end
m = mantegna(alpha,gam,n,N);
y = yang(N);
figure;
h1=histogram(m,bins,'Normalization','pdf');
set(h1,'FaceColor',[0.4 0.4 0.4],'EdgeColor',[0.4 0.4 0.4]);
hold on
h2=histogram(y,bins,'Normalization','pdf');
set(h2,'FaceColor','k','EdgeColor','k');
hold on
plot(bins,L,'LineWidth',1.5,'Color','k')
hold on
x = [-20:.1:20];
norm = normpdf(x,0,1.0);
plot(x,norm,':','LineWidth',2.5,'Color','k')
xlabel('Z')
ylabel('P(z)')
title('Alpha=1.5, Gamma=1, n=10, N=1E6')
legend('Mantegna', 'Yang', 'Levy Distribution', 'Normal Distribution')

figure(2);
semilogy(bins,L,'LineWidth',1.5,'Color','k')
hold on
semilogy(x,norm,':','LineWidth',2.5,'Color','k')
xlabel('Z')
ylabel('P(z)')
title('Normal vs Levy Distribution')
legend('Levy Distribution', 'Normal Distribution')
ylim([1E-6 1])

figure(3);
plot(bins,L,'LineWidth',1.5,'Color','k')
hold on
plot(x,norm,':','LineWidth',2.5,'Color','k')
xlabel('Z')
ylabel('P(z)')
title('Normal vs Levy Distribution')
legend('Levy Distribution', 'Normal Distribution')
ylim([0 0.5])