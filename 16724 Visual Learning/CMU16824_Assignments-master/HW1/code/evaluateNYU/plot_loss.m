close all;
clc;
clear;

fid=fopen('/home/xinshuo/tmp/10703hw1/save/training/train_scratch.log','r');
InputText=textscan(fid,'%s','delimiter',' ');
InputText = InputText{1};

loss = [];
% % for i = 1:size(InputText, 1)
for i = 1:2000*56
    temp = InputText(5*i - 1);
    loss = [loss, str2double(temp{1, 1})]; 
end
loss = loss';
x = 1:1/2000:57;
x = x(1:size(loss, 1));

figure;
plot(x, loss);
xlabel('Epoch');
ylabel('Loss');
title('Training from scratch');
print('scratch', '-depsc')