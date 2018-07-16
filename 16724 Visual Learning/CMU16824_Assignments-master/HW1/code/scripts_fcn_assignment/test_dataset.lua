require 'torch'
require 'nngraph'
local nn = require 'nn'
require 'nnx'
require 'optim'
require 'image'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'
dbg = require '../luadebugger/debugger'

-- local trainLoader = require 'dataset'

image_utils = require 'image'
-- require 'dataset'
local sanitize = require('sanitize')

torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
-- parse command-line options
-- TODO: put your path for saving models in "save" 
opt = lapp[[
  -s,--save           (default "/home/xinshuo/tmp/10703hw1/save")      subdirectory to save logs
  --saveFreq          (default 5)          save every saveFreq epochs
  -n,--network        (default "")          reload pretrained network
  -r,--learningRate   (default 0.001)      learning rate
  -b,--batchSize      (default 10)         batch size
  -m,--momentum       (default 0.9)         momentum term of adam
  -t,--threads        (default 1)           number of threads
  -g,--gpu            (default 0)          gpu to run on (default cpu)
  --scale             (default 512)          scale of images to train on  -- height and width of image
  --epochSize         (default 2000)        number of samples per epoch
  --forceDonkeys      (default 0)
  --nDonkeys          (default 1)           number of data loading threads
  --weightDecay       (default 0.0005)        weight decay
  --classnum          (default 40)    
  --classification    (default 1)
  --network_path      (default "/home/xinshuo/models/AlexNet")
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
print(opt)

opt.loadSize  = opt.scale   
-- TODO: setup the output size 
opt.labelSize = 16


opt.manualSeed = torch.random(1,10000) 
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}
opt.outDim =  opt.classnum



paths.dofile('data.lua')  -- run test
-- paths.dofile('donkey.lua')
-- trainLoader = dataLoader{split=100, sampleSize={3, 512, 512}, samplingMode = 'random'}

print('start sampling')
print(trainLoader)
-- trainLoader.imagePath
data, scalarLabels = trainLoader:sample(opt.batchSize)
