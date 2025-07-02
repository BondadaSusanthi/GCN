num=[90]
den=[1 9 18]
sys=tf(num,den)
nyquist(sys)
title('nyquist plot')
[Gm,Pm,Wcg,Wcp]=margin(num,den)
gain_margin=20*log10(Gm)
grid
phase_margin=Pm
gain_margin=Gm
gaincrossover=Wcp
phase_crossover=Wcg
