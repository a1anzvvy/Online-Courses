require 'torch'
require 'optim'
require 'pl'
require 'paths'
local dbg = require 'debugger'

local fcn = {}

local trainLogger
if opt.phase == "pre" then
  print('executing training from pre-trained model')
  trainLogger = optim.Logger(paths.concat(opt.save, 'train_pre.log'))
end

if opt.phase == "scratch" then
  print('executing training from scratch')
  trainLogger = optim.Logger(paths.concat(opt.save, 'train_scratch.log'))
end

local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
-- put the labels for each batch in targets
local targets = torch.Tensor(opt.batchSize * opt.labelSize * opt.labelSize)

local sampleTimer = torch.Timer()
local dataTimer = torch.Timer()

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

-- local criterion = nn.ClassNLLCriterion():cuda()
local criterion = nn.CrossEntropyCriterion():cuda()


local optimState = {
    learningRate = opt.learningRate,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

print(optimState)
local parameters, gradParameters = model_FCN:getParameters()

-- training function
function fcn.train(inputs_all)
  cutorch.synchronize()
  collectgarbage()

  epoch = epoch or 1
  local dataLoadingTime = dataTimer:time().real; sampleTimer:reset(); -- timers
  local dataBatchSize = opt.batchSize 

  -- TODO: check the output of the batch
  inputsCPU = inputs_all[1]
  labelsCPU = inputs_all[2]
  -- print('input size')
  -- print(inputsCPU:size())
  -- print(labelsCPU:size())
  -- dbg()
  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)

  local err, outputs
  feval = function(x) --takes point of evaluation and returns f(x) and df/dx
    model_FCN:zeroGradParameters()
    outputs = model_FCN:forward(inputs)
    -- print(outputs:size())
    -- print(model_FCN.modules[16].output:size())
    -- print(model_FCN.modules[17].output:size())
    -- print(model_FCN.modules[18].output:size())
    -- print(model_FCN.modules[19].output:size())
    -- dbg()
    err = criterion:forward(outputs, labels)
    -- model_FCN:zeroGradParameters()
    -- print(err:size())
    local gradOutputs = criterion:backward(outputs, labels)
    -- print(gradOutputs:size())
    -- dbg()
    model_FCN:backward(inputs, gradOutputs)
    return err, gradParameters
  end

  optim.sgd(feval, parameters, optimState) --stochastic gradient descent

  -- DataParallelTabel's syncParameters
  if model_FCN.needsSync then
    model_FCN:syncParameters()
  end

  loss_epoch = loss_epoch + err
  batchNumber = batchNumber + 1

  cutorch.synchronize()
  collectgarbage()
  
  print(('Epoch: [%d/%d][%d/%d]\tLoss: %.6f\tTime %.3f DataTime %.3f\tLR: %.5f'):format(epoch, opt.nEpochs, batchNumber, opt.epochSize, err, sampleTimer:time().real, dataLoadingTime, optimState.learningRate))
  trainLogger:add{['Epoch:'] = epoch, ['Iteration:'] = batchNumber, ['Loss:'] = err, ['learningRate:'] = optimState.learningRate}
  dataTimer:reset()

end


return fcn


