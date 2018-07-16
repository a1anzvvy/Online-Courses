--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
require 'struct'
require 'image'
require 'string'
require 'dataset'
-- paths.dofile('dataset.lua')
dbg = require 'debugger'
local totem = require 'totem'
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "/home/xinshuo/tmp/10703hw1/cache"
os.execute('mkdir -p ' .. cache)
local trainCache = paths.concat(cache, 'trainCache_assignment2.t7')


-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or '/home/xinshuo/tmp/10703hw1/logs'
--------------------------------------------------------------------------------------------
if not os.execute('cd ' .. opt.data) then
    os.execute('mkdir -p ' .. opt.data)
    print(('log folder is created in %s'):format(opt.data))
    -- error(("could not chdir to '%s'"):format(opt.data))
end


local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.loadSize}
local labelSampleSize = {3, opt.labelSize}

-- read the codebook (40 * 3)

local codebooktxt = '../list/codebook_40.txt' 
local codebook = torch.Tensor(40,3)
if type(opt.classification) == 'number' and opt.classification == 1 then 

  local fcode = torch.DiskFile(codebooktxt, 'r')
  for i = 1, 40 do 
    for j = 1, 3 do 
      codebook[{{i},{j}}] = fcode:readFloat()
    end
  end
  fcode:close()
end


local div_num, sub_num
div_num = 127.5
sub_num = -1


local function loadImage(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.loadSize, opt.loadSize)
   input = input * 255
   return input
end


local function loadLabel_high(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.labelSize, opt.labelSize )
   input = input * 255
   -- print(input)
   -- dbg()
   return input
end



function makeData_cls(img, label)
  -- TODO: the input label is a 3-channel real value image, quantize each pixel into classes (1 ~ 40)
  -- resize the label map from a matrix into a long vector
  -- hint: the label should be a vector with dimension of: opt.batchSize * opt.labelSize * opt.labelSize

  totem.assertStorageEq(label:size(), torch.LongStorage({opt.batchSize, 3, opt.labelSize, opt.labelSize}))
  totem.assertStorageEq(img:size(), torch.LongStorage({opt.batchSize, 3, opt.loadSize, opt.loadSize}))

  -- print(label:size())
  local label_tmp = label:transpose(2, 1)   -- permute before reshape
  -- print(label_tmp:size())
  local label_vector = torch.reshape(label_tmp, 3, opt.batchSize * opt.labelSize * opt.labelSize)   -- 3x2560
  local res = torch.zeros(40, opt.batchSize * opt.labelSize * opt.labelSize)                       -- 40x2560
  -- print(label_vector:transpose(1,2)[1])
  -- print(codebook)
  res:addmm(codebook, label_vector) 
  -- print(res:transpose(1,2)[1])
  local max_vector, label = torch.max(res, 1)    -- find the argmax for the label
  -- print(label[1][1])
  -- dbg()
  label = torch.reshape(label, opt.batchSize * opt.labelSize * opt.labelSize)    -- TODO: check the correspondence
  totem.assertStorageEq(label:size(), torch.LongStorage({opt.batchSize * opt.labelSize * opt.labelSize}))
  return {img, label}
end


function makeData_cls_pre(img, label)
  -- TODO: almost same as makeData_cls, need to convert img from RGB to BGR for caffe pre-trained model
  local data = makeData_cls(img, label)
  -- print(data[1]:size())
  -- print(data[2]:size())
  img = data[1]
  label = data[2]
  -- img_dis = img[1]
  -- print(img_dis:size())
  -- img_dis = img_dis:transpose(2,1)
  -- print(img_dis:size())
  -- img_dis = img_dis:transpose(3,2)
  -- print(img_dis:size())
  -- image.display(img_dis)
  -- image.toDisplayTensor(img_dis)
  -- print(img_dis)
  -- image.save('test.jpg', img_dis)
  -- dbg()
  -- print(img:size())
  -- dbg()
  -- print(img[1][1][1][1])
  -- print(img[1][2][1][1])
  -- print(img[1][3][1][1])
  -- dbg()
  local img_tmp = img:index(2, torch.LongTensor{3, 2, 1})  
  
  -- print(img_tmp[1][1][1][1])
  -- print(img_tmp[1][2][1][1])
  -- print(img_tmp[1][3][1][1])
  -- dbg()
  img = img_tmp
  -- print(img:size())
  -- print(label:size())
  -- print(label)
  -- dbg()
  return {img, label}
end



--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, imgpath, lblpath)
   collectgarbage()
   local img = loadImage(imgpath)
   local label = loadLabel_high(lblpath)
   img:add( - 127.5 )   -- subtract the mean
   label:div(div_num)  
   label:add(sub_num)   -- normalize the label

   return img, label

end

-- print('I am here')
--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {3, sampleSize[2], sampleSize[2]}
   trainLoader.labelSampleSize = {3, labelSampleSize[2], labelSampleSize[2]}
   print(trainLoader)
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, 'train')},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
      labelSampleSize = {3, labelSampleSize[2], labelSampleSize[2]},
      -- samplingMode = 'balanced',
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end

collectgarbage()



