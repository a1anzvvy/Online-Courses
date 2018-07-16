close all;
clear;
clc;

type = 'conv1_scratch';
list = dir(type);
number_row = 16;
conv1_list = {};
for i = 1:size(list, 1)
    index = strfind(list(i).name, 'jpeg');
    if ~isempty(index)
        conv1_list = [conv1_list; list(i).name];
    end
end
assert(mod(size(conv1_list, 1), number_row) == 0, 'Concatenation failed! Please change number of images in each row');

% image_id = 523;
% predict = imread(sprintf('/home/xinshuo/dataset/NYU/res/rgb_%06d.jpg', image_id));
% original = imread(sprintf('/home/xinshuo/dataset/NYU/croptest/rgb_%06d.jpg', image_id));

% predict = imresize(predict, [128, 128]);
% original = imresize(original, [128, 128]);
% concate = [predict, original];

image_total = [];
for i = 1:size(conv1_list, 1)/number_row
    image_tmp = [];
    for j = 1:number_row
        image_path = sprintf('%s/%s', type, conv1_list{j + (i-1) * number_row});
        image = imread(image_path);
        image_tmp = [image_tmp, image];
    end
    image_total = [image_total; image_tmp];

end

figure;
imshow(image_total);
% figure;
% imshow(original);
print(type, '-depsc')