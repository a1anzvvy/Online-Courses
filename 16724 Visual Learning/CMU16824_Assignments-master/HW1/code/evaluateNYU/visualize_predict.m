close all;
clear;
clc;

image_id = [926, 861, 821, 523, 1144, 154, 430, 464, 465, 470];
image_total = [];
for i = 1:10
    predict = imread(sprintf('/home/xinshuo/dataset/NYU/res/rgb_%06d.jpg', image_id(i)));
    original = imread(sprintf('/home/xinshuo/dataset/NYU/croptest/rgb_%06d.jpg', image_id(i)));
    predict = imresize(predict, [128, 128]);
    original = imresize(original, [128, 128]);
    concate = [predict; original];
    image_total = [image_total, concate];
end



figure;
imshow(image_total);
% figure;
% imshow(original);
print('prediction_scratch', '-depsc')