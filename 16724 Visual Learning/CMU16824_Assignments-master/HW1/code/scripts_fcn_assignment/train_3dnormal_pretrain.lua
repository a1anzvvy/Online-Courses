require 'torch'
require 'nngraph'
local nn = require 'nn'
require 'nnx'
require 'optim'
require 'cutorch'
require 'image'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'
local gm = require 'graphicsmagick'
local dbg = require 'debugger'
-- image_utils = require 'image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end



----------------------------------------------------------------------
-- parse command-line options
-- TODO: put your path for saving models in "save" 
opt = lapp[[
  -s,--save           (default "/home/xinshuo/tmp/10703hw1/save/training2/")      subdirectory to save logs
  --saveFreq          (default 1)          save every saveFreq epochs
  -n,--network        (default "")          reload pretrained network
  -r,--learningRate   (default 0.001)      learning rate
  -b,--batchSize      (default 10)         batch size
  -m,--momentum       (default 0.9)         momentum term of adam
  -t,--threads        (default 10)           number of threads
  -g,--gpu            (default 0)          gpu to run on (default cpu)
  --scale             (default 512)          scale of images to train on  -- height and width of image
  --epochSize         (default 2000)        number of samples per epoch
  --nEpochs           (default 30)          number of epoch
  --forceDonkeys      (default 0)
  --nDonkeys          (default 10)           number of data loading threads
  --weightDecay       (default 0.0005)        weight decay
  --classnum          (default 40)    
  --classification    (default 1)
  --network_path      (default "/home/xinshuo/models/AlexNet")
  --phase             (default "pre")
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





if opt.network == '' then
  ---------------------------------------------------------------------
  -- TODO: load pretrain model and add some layers, let's name it as model_FCN for the sake of simplicity
  -- hint: you might need to add large padding in conv1 (perhaps around 100ish? )
  -- hint2: use ReArrange instead of Reshape or View
  -- hint3: you might need to skip the dropout and softmax layer in the model
  print('pretrained model is loaded')
  model_FCN = torch.load(opt.network_path)
  model_FCN.modules[1].padW = 100
  model_FCN.modules[1].padH = 100
  model_FCN:remove()
  model_FCN:remove()
  model_FCN:remove(16)
  -- model_FCN:remove(18)
  -- model_FCN:remove(20)

  -- model_FCN:remove()
  -- model_FCN:remove()

  model_FCN:add(nn.SpatialConvolution(4096, opt.classnum, 1, 1, 1, 1))
  model_FCN:add(nn.Transpose({2,3},{3,4}))
  model_FCN:add(nn.View(-1, opt.classnum))
  -- model_FCN:add(nn.LogSoftMax())

  model_FCN.modules[22].weight:normal(0.0, 0.01)
  model_FCN.modules[22].bias:fill(0)
else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_FCN = tmp
end



optimState = {
    learningRate = opt.learningRate,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}




local sanitize = require('sanitize')
-- print networks
print('fcn network:')
print(model_FCN)
-- print()
model_FCN:cuda()
print('Network is in the GPU')
-- TODO: retrieve parameters and gradients
-- print(model_FCN.modules[20].bias)

-- dbg()
require 'dataset'
paths.dofile('data.lua')  -- parse
print('finishing parsing data file')
-- paths.dofile('donkey.lua')  -- parse
-- print('finishing parsing donkey file')
-- paths.dofile('dataset.lua')
-- print('finishing parsing data file')
-- print(model_FCN.modules[1].weight)
-- print(model_FCN.modules[16].bias)
-- dbg()
-- trainLoader = dataLoader{split=100, sampleSize={3, 512, 512}, samplingMode = 'random'}

-- dbg()

local fcn = require 'fcn_train_cls'   -- get the fcn training module

local function train()
  print('\n<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. optimState.learningRate .. ', momentum = ' .. optimState.momentum .. ']')
  
  model_FCN:training()
  batchNumber = 0
  for i=1,opt.epochSize do
    donkeys:addjob(
      function()
        return makeData_cls_pre(trainLoader:sample(opt.batchSize))
      end,
      fcn.train)
  end
  donkeys:synchronize()
  cutorch.synchronize()
end


print('start training\n')
epoch = 1

-- training loop
while true do
  -- train/test
  loss_epoch = 0
  train()

  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, string.format('fcn_%d_pretrain.net',epoch))
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print(('Total Epoch Loss: %.8f\t<trainer> saving network to %s'):format(loss_epoch, filename))
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
