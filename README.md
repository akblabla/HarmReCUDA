# HarmReCUDA

The program reads a in.mat file in the same directory of the executable, or the path specified by the optional launch parameter. The in.mat file must contain the following variables:

data: a sampleCount X stackCount matrix containing all the data stacks.
fMax: a double containing the maximum frequency of the search grid.
fMin: a double containing the minimum frequency of the search grid.
fRes: an integer containing the amount of frequencies in the search grid.
fs: a double containing the sample rate.
harmonics: a row vector containing the harmonic indexes of the harmonics to be removed.

