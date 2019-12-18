clear all
clc
close all
%mex -setup cpp
mex -Llibfolder HarmReMex.cpp

%mex '-LC:Documents\Bachelor\Projects\bin\win64\Debug'-HarmReCUDA HarmReMex.cpp