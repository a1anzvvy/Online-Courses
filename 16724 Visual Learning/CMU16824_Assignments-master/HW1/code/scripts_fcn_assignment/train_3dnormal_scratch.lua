require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require 'cutorch'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'
image_utils = require 'image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end


----------------------------------------------------------------------
-- parse command-line options
-- TODO: put your path for saving models in "save" 
opt = lapp[[
  -s,--save          (default "/home/xinshuo/tmp/10703hw1/save/training/")      subdirectory to save logs
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -r,--learningRate  (default 0.01)      learning rate
  -b,--batchSize     (default 10)         batch size
  -m,--momentum      (default 0.9)         momentum term of adam
  -t,--threads       (default 10)           number of threads
  -g,--gpu           (default 0)          gpu to run on (default cpu)
  --scale            (default 512)          scale of images to train on
  --epochSize        (default 2000)        number of samples per epoch
  --forceDonkeys     (default 0)
  --nDonkeys         (default 10)           number of data loading threads
  --weightDecay      (default 0.0005)        weight decay
  --classnum         (default 40)    
  --classification   (default 1)
  --phase            (default "scratch")
  --nEpochs          (default 30)          number of epoch
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
print(opt)

opt.loadSize  = opt.scale 
-- TODO: setup the output size 
opt.labelSize = 32


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


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.01)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


if opt.network == '' then
  ---------------------------------------------------------------------
  -- TODO: write your own networks, let's name it as model_FCN for the sake of simplicity
  -- hint: you might need to add large padding in conv1 (perhaps around 100ish? )
  -- hint2: use ReArrange instead of Reshape or View
  model_FCN = nn.Sequential()
  model_FCN:add(nn.SpatialConvolution(3, 96, 11, 11, 4, 4, 68, 68))
  model_FCN:add(nn.SpatialBatchNormalization(96))
  model_FCN:add(nn.ReLU())
  model_FCN:add(nn.SpatialMaxPooling(3, 3, 2, 2))
  model_FCN:add(nn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2))
  model_FCN:add(nn.SpatialBatchNormalization(256))
  model_FCN:add(nn.ReLU())
  model_FCN:add(nn.SpatialMaxPooling(3, 3, 2, 2))
  model_FCN:add(nn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1))
  model_FCN:add(nn.SpatialBatchNormalization(384))
  model_FCN:add(nn.ReLU())
  -- model_FCN:add(nn.SpatialMaxPooling(3, 3, 2, 2))
  model_FCN:add(nn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1))
  model_FCN:add(nn.SpatialBatchNormalization(384))
  model_FCN:add(nn.ReLU())
  model_FCN:add(nn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1))
  model_FCN:add(nn.SpatialBatchNormalization(256))
  model_FCN:add(nn.ReLU())
  model_FCN:add(nn.SpatialMaxPooling(3, 3, 2, 2))
  model_FCN:add(nn.SpatialConvolution(256, 1024, 6, 6, 1, 1, 1, 1))
  model_FCN:add(nn.SpatialBatchNormalization(1024))
  model_FCN:add(nn.ReLU())
  model_FCN:add(nn.SpatialFullConvolution(1024, 512, 4, 4, 2, 2, 1, 1))
  model_FCN:add(nn.SpatialBatchNormalization(512))
  model_FCN:add(nn.ReLU())
  model_FCN:add(nn.SpatialConvolution(512, 40, 3, 3, 1, 1, 1, 1))
  model_FCN:add(nn.Transpose({2,3},{3,4}))
  model_FCN:add(nn.View(-1, opt.classnum))
  -- model_FCN:add(nn.LogSoftMax())

  model_FCN:apply(weights_init)

else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_FCN = tmp.FCN
end

-- print networks
print('fcn network:')
print(model_FCN)
model_FCN:cuda()

paths.dofile('data.lua')  -- parse
print('finishing parsing data file')

local sanitize = require('sanitize')


local fcn = require 'fcn_train_cls'   -- get the fcn training module

-- TODO: loss function
-- TODO: retrieve parameters and gradients
-- TODO: setup dataset, use data.lua
-- TODO: setup training functions, use fcn_train_cls.lua
-- TODO: convert model and loss function to cuda


local optimState = {
    learningRate = opt.learningRate,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}


local function train()
   print('\n<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. optimState.learningRate .. ', momentum = ' .. optimState.momentum .. ']')
  
   model_FCN:training()
   batchNumber = 0
   for i=1,opt.epochSize do
      donkeys:addjob(
         function()
            return makeData_cls(trainLoader:sample(opt.batchSize))
         end,
         fcn.train)
   end
   donkeys:synchronize()
   cutorch.synchronize()

end

epoch = 1
-- training loop
while true do
  -- train/test
  loss_epoch = 0
  train()

  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, string.format('fcn_%d_scratch.net',epoch))
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, { FCN = sanitize(model_FCN), opt = opt})
  end

  epoch = epoch + 1

  -- plot errors
  if opt.plot  and epoch and epoch % 1 == 0 then
    torch.setdefaulttensortype('torch.FloatTensor')

    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
