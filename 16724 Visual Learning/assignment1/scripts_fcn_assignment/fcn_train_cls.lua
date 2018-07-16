require 'torch'
require 'optim'
require 'pl'
require 'paths'

local fcn = {}

local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
-- put the labels for each batch in targets
local targets = torch.Tensor(opt.batchSize * opt.labelSize * opt.labelSize)

local sampleTimer = torch.Timer()
local dataTimer = torch.Timer()


-- training function
function fcn.train(inputs_all)
  cutorch.synchronize()
  epoch = epoch or 1
  local dataLoadingTime = dataTimer:time().real; sampleTimer:reset(); -- timers
  local dataBatchSize = opt.batchSize 

  -- TODO: implemnet the training function 

  batchNumber = batchNumber + 1
  cutorch.synchronize(); collectgarbage();
  print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f'):format(epoch, batchNumber, opt.epochSize, sampleTimer:time().real, dataLoadingTime))
  dataTimer:reset()

end


return fcn


