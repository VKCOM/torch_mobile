local Module = torch.class('nn.Module')

function Module:__init()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
end

function Module:parameters()
   if self.weight and self.bias then
      return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
   elseif self.weight then
      return {self.weight}, {self.gradWeight}
   elseif self.bias then
      return {self.bias}, {self.gradBias}
   else
      return
   end
end

function Module:updateOutput(input)
   return self.output
end

function Module:forward(input)
   return self:updateOutput(input)
end

function Module:backward(input, gradOutput, scale)
   scale = scale or 1
   self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput, scale)
   return self.gradInput
end

function Module:backwardUpdate(input, gradOutput, lr)
   self:updateGradInput(input, gradOutput)
   self:accUpdateGradParameters(input, gradOutput, lr)
   return self.gradInput
end

function Module:updateGradInput(input, gradOutput)
   return self.gradInput
end

function Module:accGradParameters(input, gradOutput, scale)
end

function Module:accUpdateGradParameters(input, gradOutput, lr)
   local gradWeight = self.gradWeight
   local gradBias = self.gradBias
   self.gradWeight = self.weight
   self.gradBias = self.bias
   self:accGradParameters(input, gradOutput, -lr)
   self.gradWeight = gradWeight
   self.gradBias = gradBias
end

function Module:sharedAccUpdateGradParameters(input, gradOutput, lr)
   if self:parameters() then
      self:zeroGradParameters()
      self:accGradParameters(input, gradOutput, 1)
      self:updateParameters(lr)
   end
end

function Module:zeroGradParameters()
   local _,gradParams = self:parameters()
   if gradParams then
      for i=1,#gradParams do
         gradParams[i]:zero()
      end
   end
end

function Module:updateParameters(learningRate)
   local params, gradParams = self:parameters()
   if params then
      for i=1,#params do
         params[i]:add(-learningRate, gradParams[i])
      end
   end
end

function Module:share(mlp, ...)
   local arg = {...}
   for i,v in ipairs(arg) do
      if self[v] ~= nil then
         self[v]:set(mlp[v])
         self.accUpdateGradParameters = self.sharedAccUpdateGradParameters
         mlp.accUpdateGradParameters = mlp.sharedAccUpdateGradParameters
      end
   end
   return self      
end

function Module:clone(...)
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   if select('#',...) > 0 then
      clone:share(self,...)
   end
   return clone
end

function Module:type(type)
   -- find all tensors and convert them
   for key,param in pairs(self) do
      if torch.typename(param) and torch.typename(param):find('torch%..+Tensor') then
         self[key] = param:type(type)
      end
   end
   -- find submodules in classic containers 'modules'
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:type(type)
      end
   end
   return self
end

function Module:float()
   return self:type('torch.FloatTensor')
end

function Module:double()
   return self:type('torch.DoubleTensor')
end

function Module:cuda()
   return self:type('torch.CudaTensor')
end

function Module:reset()
end

function Module:getParameters()
   -- get parameters
   local parameters,gradParameters = self:parameters()

   local function storageInSet(set, storage)
      local storageAndOffset = set[torch.pointer(storage)]
      if storageAndOffset == nil then
          return nil
      end
      local storage, offset = unpack(storageAndOffset)
      return offset
   end

   -- this function flattens arbitrary lists of parameters,
   -- even complex shared ones
   local function flatten(parameters)
      local Tensor = parameters[1].new

      local storages = {}
      local nParameters = 0
      for k = 1,#parameters do
         local storage = parameters[k]:storage()
         if not storageInSet(storages, storage) then
            storages[torch.pointer(storage)] = {storage, nParameters}
            nParameters = nParameters + storage:size()
         end
      end
      
      local flatParameters = Tensor(nParameters):fill(1)
      local flatStorage = flatParameters:storage()

      for k = 1,#parameters do
         local storageOffset = storageInSet(storages, parameters[k]:storage())
         parameters[k]:set(flatStorage,
                           storageOffset + parameters[k]:storageOffset(),
                           parameters[k]:size(),
                           parameters[k]:stride())
         parameters[k]:zero()
      end

      local cumSumOfHoles = flatParameters:float():cumsum(1)
      local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
      local flatUsedParameters = Tensor(nUsedParameters)
      local flatUsedStorage = flatUsedParameters:storage()

      for k = 1,#parameters do
         local offset = cumSumOfHoles[parameters[k]:storageOffset()]
         parameters[k]:set(flatUsedStorage,
         parameters[k]:storageOffset() - offset,
         parameters[k]:size(),
         parameters[k]:stride())
      end

      for _, storageAndOffset in pairs(storages) do
         local k, v = unpack(storageAndOffset)
         flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
      end
      if cumSumOfHoles:sum() == 0 then
         flatUsedParameters:copy(flatParameters)
      else
         for k = 1,flatUsedParameters:nElement() do
            flatUsedParameters[k] = flatParameters[k+cumSumOfHoles[k]]
         end
      end
      return flatUsedParameters
   end

   -- flatten parameters and gradients
   local flatParameters = flatten(parameters)
   local flatGradParameters = flatten(gradParameters)

   -- return new flat vector that contains all discrete parameters
   return flatParameters, flatGradParameters
end

-- returns a list of modules
function Module:listModules()
   local function tinsert(to, from)
      if torch.type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   -- include self first
   local modules = {self}
   if self.modules then
      for i=1,#self.modules do
         local modulas = self.modules[i]:listModules()
         if modulas then
            tinsert(modules,modulas)
         end
      end
   end
   return modules
end

function Module:apply(callback)
    callback(self)

    if self.modules then
        for _, module in ipairs(self.modules) do
            module:apply(callback)
        end
    end
end

function Module:findModules(typename, container)
  container = container or self
  local nodes = {}
  local containers = {}
  local mod_type = torch.typename(self)
  if mod_type == typename then
    nodes[#nodes+1] = self
    containers[#containers+1] = container
  end
  -- Recurse on nodes with 'modules'
  if (self.modules ~= nil) then
    if (torch.type(self.modules) == 'table') then
      for i = 1, #self.modules do
        local child = self.modules[i]
        local cur_nodes, cur_containers =
          child:findModules(typename, self)
        assert(#cur_nodes == #cur_containers,
          'Internal error: incorrect return length')  -- This shouldn't happen
        -- add the list items from our child to our list (ie return a
        -- flattened table of the return nodes).
        for j = 1, #cur_nodes do
          nodes[#nodes+1] = cur_nodes[j]
          containers[#containers+1] = cur_containers[j]
        end
      end
    end
  end
  return nodes, containers
end

function Module:evaluate()
  self.train = false
end

function Module:training()
  self.train = true
end

-- garbage collection enforcement
function Module:freeTensors(mod, of)
  local of = of or false
  for k, v in pairs( mod ) do
    if torch.isTensor(v) and v:storage() then
      if k~="output" or of then
        mod[k] = nil
      end
    end
  end
  collectgarbage()
end

function Module:freeInputs()
  local cnt = 0
  for k, v in pairs (self.inputs) do
    v:storage():resize(0)
    self.inputs[k] = nil
    cnt = cnt + 1
  end
  print(string.format("%s inputs have been destroyed", cnt))
end

-- refernece counter based on optimize-net
function Module:freeTensorsOptGC(mod)
  if mod.refCounter == nil then
    print('Can not free Tensors')
  else
    for k, v in pairs( mod ) do
      if torch.isTensor(v) and v:storage() and k~="output" then
        local ptr = torch.pointer(v:storage())
        if mod.refCounter[ptr] ~= nil then
          mod.refCounter[ptr] = mod.refCounter[ptr] - 1
          if mod.refCounter[ptr] == 0 then
              mod[k] = nil
          end
        end -- self.refCounter[ptr] ~= nil
      end -- torch.isTensor(v) and v:storage()
    end -- for k, v in pairs( mod )
  end -- mod.refCounter == nil
  collectgarbage()
end

function Module:__call__(input, gradOutput)
   self:forward(input)
   if gradOutput then
      self:backward(input, gradOutput)
      return self.output, self.gradInput
   else
      return self.output
   end
end
