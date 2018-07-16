require 'torch'
require 'nngraph'
local nn = require 'nn'
require 'cunn'
require 'nnx'
require 'optim'
require 'cutorch'
require 'image'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'
local gm = require 'graphicsmagick'


local sanitize = require('sanitize')
-- local dbg = require 'debugger'
-- image_utils = require 'image'
-- ok, disp = pcall(require, 'display')
-- if not ok then print('display not found. unable to plot') end




local function visualize(conv1)
  -- local conv1 = model.modules[1].weight

  local conv1_cpu = torch.DoubleTensor()
  conv1_cpu = conv1_cpu:resize(conv1:size()):copy(conv1)
  -- local min = conv1_cpu:min()
  conv1_cpu = conv1_cpu:add(-conv1_cpu:min())
  conv1_cpu = conv1_cpu:div(conv1_cpu:max()-conv1_cpu:min())

  for ii = 1, 96 do
    local save_conv1_path = 'conv1_' .. ii .. '.jpeg'
    local myimage = gm.Image(conv1_cpu[ii], 'RGB', 'DHW')
    -- print(type(save_conv1_path))
    -- print(conv1[ii]:size())
    myimage:save(save_conv1_path)
    -- print(image.minmax(conv1_cpu[1]))
  end

end




model = torch.load('/home/xinshuo/models/AlexNet')
print(model)
model = model:cuda()
print(model)
visualize(model.modules[1].weight)