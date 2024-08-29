%nproc=8
%Mem=4GB
#p BLYP/def2svp scan nosymm

OH molecule PES scan

0 1
H
O 1 R1
H 2 R2 1 A1

R2 = 0.5 S 10 0.1
R1 = 0.9667850687768991
A1 = 104