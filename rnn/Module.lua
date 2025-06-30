local _ = require 'moses'
local Module = nn.Module

-- You can use this to manually forget past memories in AbstractRecurrent instances
function Module:forget()
   if self.modules then
      for i,module in ipairs(self.modules) do
         module:forget()
      end
   end
   return self
end

-- Used by nn.Sequencers
function Module:remember(remember)
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:remember(remember)
      end
   end
   return self
end

function Module:maskZero(v1)
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:maskZero(v1)
      end
   end
   return self
end

function Module:setZeroMask(zeroMask)
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:setZeroMask(zeroMask)
      end
   end
end

function Module:stepClone(shareParams, shareGradParams)
   return self:sharedClone(shareParams, shareGradParams, true)
end

-- calls setOutputStep on all component AbstractRecurrent modules
-- used by Recursor() after calling stepClone.
-- this solves a very annoying bug...
function Module:setOutputStep(step)
   if self.modules then
      for i,module in ipairs(self.modules) do
         module:setOutputStep(step)
      end
   end
end

function Module:sparseParameters()
   return self:parameters()
end

function Module:updateParameters(learningRate)
   -- sparse params can have different learningRate scales per param
   local params, gradParams, scales = self:sparseParameters()
   if params then
      for i,param in pairs(params) do -- pairs for sparse params
         local scale = scales and scales[i] or 1
         param:add(-learningRate*scale, gradParams[i])
      end
   end
end

function Module:zeroGradParameters()
   local _,gradParams = self:sparseParameters()
   if gradParams then
      for i,gradParam in pairs(gradParams) do -- pairs for sparse params
         gradParam:zero()
      end
   end
end

------------------------ clone and type --------------------------------

Module.dpnn_parameters = {'weight', 'bias'}
Module.dpnn_gradParameters = {'gradWeight', 'gradBias'}

-- efficient version of :
-- clone = self:clone()
-- clone:share(self, paramNames, gradParamNames)
-- Note that this method is the very bane of my existence.
-- I have worked on it too many times...
function Module:sharedClone(shareParams, shareGradParams, stepClone)
   shareParams = (shareParams == nil) and true or shareParams
   shareGradParams = (shareGradParams == nil) and true or shareGradParams

   if stepClone and self.dpnn_stepclone then
      -- this is for AbstractRecurrent modules
      return self
   end

   local pointers = {} -- to params/gradParams (dont clone params/gradParams)
   local scdone = {}

   -- 1. remove all params/gradParams
   local function recursiveRemove(obj) -- remove modules
      local moduleTree
      local isTable = type(obj) == 'table'
      if torch.isTypeOf(obj, 'nn.Module') then
         assert(isTable)
         if stepClone and obj.dpnn_stepclone then
            -- this is for AbstractRecurrent modules (in rnn)
            moduleTree = obj
            obj = nil
            isTable = false
         elseif obj.dpnn_sharedClone then
            -- allow to use a custom sharedClone method on one module
            moduleTree = obj
            obj = nil
            isTable = false
         elseif scdone[torch.pointer(obj)] then
            moduleTree = scdone[torch.pointer(obj)]
         else
            -- remove the params, gradParams. Save for later.
            local params = {}

            if shareParams then
               for i,paramName in ipairs(obj.dpnn_parameters) do
                  local param = obj[paramName]
                  if param then
                     params[paramName] = param
                     obj[paramName] = nil
                     if torch.isTensor(param) and param.storage and param:storage() then
                        pointers[torch.pointer(param:storage():data())] = true
                     end
                  end
               end
            end

            if shareGradParams then
               for i,paramName in ipairs(obj.dpnn_gradParameters) do
                  local gradParam = obj[paramName]
                  if gradParam then
                     params[paramName] = gradParam
                     obj[paramName] = nil
                     if torch.isTensor(gradParam) and gradParam.storage and gradParam:storage() then
                        pointers[torch.pointer(gradParam:storage():data())] = true
                     end
                  end
               end
            end

            -- find all obj.attribute tensors that share storage with the shared params
            for paramName, param in pairs(obj) do
               if torch.isTensor(param) and param:storage() then
                  if pointers[torch.pointer(param:storage():data())] then
                     params[paramName] = param
                     obj[paramName] = nil
                  end
               end
            end

            moduleTree = params

            scdone[torch.pointer(obj)] = moduleTree

            for k,v in pairs(obj) do
               moduleTree[k], obj[k] = recursiveRemove(v)
            end

         end
      elseif isTable then
         if scdone[torch.pointer(obj)] then
            moduleTree = scdone[torch.pointer(obj)]
         else
            assert(not moduleTree)
            moduleTree = {}
            for k,v in pairs(obj) do
               moduleTree[k], obj[k] = recursiveRemove(v)
            end
            scdone[torch.pointer(obj)] = moduleTree
         end

      end

      return moduleTree, obj
   end

   local moduleTree, original = recursiveRemove(self)
   assert(original)

   -- 2. clone everything but parameters, gradients and modules (removed above)

   local clone = self:clone()

   -- 3. add back to self/clone everything that was removed in step 1

   local function recursiveSet(clone, original, moduleTree)
      assert(clone)
      assert(original)
      if scdone[torch.pointer(original)] then
         for k,param in pairs(moduleTree) do
            if torch.isTypeOf(param,'nn.Module') then
               if param.dpnn_sharedClone then
                  -- Call the custom sharedClone
                  clone[k] = param:dpnn_sharedClone()
               else
                  -- AbstractRecurrent instances branch here with stepClone = true
                  clone[k] = param
               end
               original[k] = param
            elseif torch.isTensor(param) then
               if param.storage then
                  clone[k] = param.new():set(param)
                  original[k] = param
               else -- for torch.MultiCudaTensor
                  clone[k] = param
                  original[k] = param
               end
            elseif type(param) == 'table' then
               recursiveSet(clone[k], original[k], param)
            end
         end
         scdone[torch.pointer(original)] = nil
      end

   end

   recursiveSet(clone, self, moduleTree)

   return clone
end

-- we override this method such that hidden modules
-- will be included in the getParameters call.
-- Hidden modules are common for recurrent modules that
-- have internal references to modules that share parameters
-- with the main modules.
-- These must also be included in the getParameters() call in order
-- to maintain shared storage for tensors.
function Module:getParameters()

   local con = nn.Container()
   con:add(self)

   -- recursive get all modules (modules, sharedclones, etc.)
   local function recursiveGetModules(tbl)
      for k,m in pairs(tbl) do
         if torch.isTypeOf(m, 'nn.Module') then
            if not m.dpnn_getParameters_found then
               con:add(m)
               m.dpnn_getParameters_found = true
               recursiveGetModules(m)
            end
         elseif torch.type(m) == 'table' then
            recursiveGetModules(m)
         end
      end
   end

   recursiveGetModules(self)

   for i,m in ipairs(con.modules) do
      m.dpnn_getParameters_found = nil
   end

   -- get ALL parameters
   local parameters,gradParameters = con:parameters()
   return Module.flatten(parameters), Module.flatten(gradParameters)
end

----------------------- for training -----------------------------

-- useful to get the output size
-- I chose this method name because it is less likely to be overriden.
function Module:outside(insize)
   local input
   if torch.type(insize) == 'table' then
      input = torch.randn(table.unpack(insize))
   else
      input = torch.randn(insize)
   end
   local output = self:updateOutput(input)
   return output:size()
end

-- for those interested in implementing the visitor design pattern
function Module:accept(visitor)
   visitor:visit(self)
end

-- Can be used as a regularizer instead of weight decay
-- Assumes that parameters are arranged (output dim x ... x input dim)
function Module:maxParamNorm(maxOutNorm, maxInNorm)
   -- this allows each module to set its own max[Out,In]Norm
   maxOutNorm = self.maxOutNorm or maxOutNorm
   maxInNorm = self.maxInNorm or maxInNorm
   if not (maxOutNorm or maxInNorm) then
      return
   end

   if self.modules then
      for i,module in ipairs(self.modules) do
         module:maxParamNorm(maxOutNorm, maxInNorm)
      end
   else
      local params = self:parameters()
      if not params or gradParams then
         return
      end
      for k,param in pairs(params) do -- pairs for sparse params
         -- By default, only affects non-1D params.
         if param:dim() > 1 then
            if maxOutNorm and maxOutNorm > 0 then
               -- rows feed into output neurons
               param:renorm(2, 1, maxOutNorm)
            end
            if maxInNorm and maxInNorm > 0 then
               -- cols feed out from input neurons
               param:renorm(2, param:dim(), maxInNorm)
            end
         end
      end
   end
end


-- set the maximum number of backpropagation through time (BPTT) time-steps
function Module:maxBPTTstep(rho)
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:maxBPTTstep(rho)
      end
   end
end

function Module:setStartState(startState)
   if self.modules then
      if torch.type(startState) == 'table' then
         for i, module in ipairs(self.modules) do
            module:setStartState(startState[i])
         end
      else
         -- empties the start states
         assert(startState == nil, "Expecting table or nil for arg 1")
         for i, module in ipairs(self.modules) do
            module:setStartState()
         end
      end
   end
end

function Module:getHiddenState(step)
   if self.modules then
      local hiddenState = {}
      for i, module in ipairs(self.modules) do
         hiddenState[i] = module:getHiddenState(step)
      end
      return hiddenState
   end
end

function Module:setHiddenState(step, hiddenState)
   if self.modules then
      assert(torch.type(hiddenState) == 'table')
      for i, module in ipairs(self.modules) do
         module:setHiddenState(step, hiddenState[i])
      end
   end
end

function Module:getGradHiddenState(step)
   if self.modules then
      local gradHiddenState = {}
      for i, module in ipairs(self.modules) do
         gradHiddenState[i] = module:getGradHiddenState(step)
      end
      return gradHiddenState
   end
end

function Module:setGradHiddenState(step, gradHiddenState)
   if self.modules then
      assert(torch.type(gradHiddenState) == 'table')
      for i, module in ipairs(self.modules) do
         module:setGradHiddenState(step, gradHiddenState[i])
      end
   end
end

-- Similar to maxParamNorm, but norm is global to Module for which
-- this is called. Unless moduleLocal is true, in which case, the
-- norm constraint is applied to the norm of all parameters in each
-- component (non-container) module.
function Module:gradParamClip(cutoffNorm, moduleLocal)
   -- this allows each module to set its own cutoffNorm
   cutoffNorm = self.cutoffNorm or cutoffNorm
   if cutoffNorm <= 0 then
      return
   end
   if self.moduleLocal ~= nil then
      moduleLocal = self.moduleLocal
   end

   local norm = 0
   if moduleLocal and self.modules then
      for i,module in ipairs(self.modules) do
         norm = norm + math.pow(module:gradParamClip(cutoffNorm, moduleLocal), 2)
      end
      norm = math.sqrt(norm)
   else
      local params, gradParams = self:parameters()
      if not (params and gradParams) then
         return norm
      end
      for k,gradParam in pairs(gradParams) do -- pairs for sparse params
         if torch.type(gradParam) == 'torch.CudaTensor' then
            cutorch.withDevice(gradParam:getDevice(), function() -- support multi-device models
               norm = norm + math.pow(gradParam:norm(),2)
            end)
         else
            norm = norm + math.pow(gradParam:norm(),2)
         end
      end
      norm = math.sqrt(norm)
      if norm > cutoffNorm then
         -- rescale gradParams to obtain desired cutoffNorm
         for k,gradParam in pairs(gradParams) do
            if torch.type(gradParam) == 'torch.CudaTensor' then
               cutorch.withDevice(gradParam:getDevice(), function() -- support multi-device models
                  gradParam:mul(cutoffNorm/norm)
               end)
            else
               gradParam:mul(cutoffNorm/norm)
            end
         end
      end
   end
   return norm
end

-- Adds weight decay constraint on params with dims > 2 (default).
-- TODO : allow inplace weightDecay (before calling accUpdateGradParameters)
function Module:weightDecay(wdFactor, wdMinDim)
   -- this allows each module to set its own hyper-parameters
   wdFactor = self.wdFactor or wdFactor
   if wdFactor <= 0 then
      return
   end
   wdMinDim = self.wdMinDim or wdMinDim or 2

   if self.modules then
      for i,module in ipairs(self.modules) do
         module:weightDecay(wdFactor, wdMinDim)
      end
   else
      local params, gradParams = self:parameters()
      if not (params and gradParams) then
         return
      end

      for i,param in pairs(params) do -- pairs for sparse params
         if param:dim() >= wdMinDim then
            gradParams[i]:add(wdFactor, param)
         end
      end
   end
end

function Module:momentumGradParameters()
   if (not self.momGradParams) or _.isEmpty(self.momGradParams) then
      local params, gradParams = self:parameters()
      if not gradParams or _.isEmpty(gradParams) then
         return
      end
      self.momGradParams = {}
      for i,gradParam in pairs(gradParams) do
         if torch.type(gradParam) == 'torch.CudaTensor' then
            cutorch.withDevice(gradParam:getDevice(), function() -- support multi-device models
               self.momGradParams[i] = gradParam.new():resizeAs(gradParam):copy(gradParam)
            end)
         else
            self.momGradParams[i] = gradParam.new():resizeAs(gradParam):copy(gradParam)
         end
      end
   end
   return self.momGradParams
end

-- uses momentum learning to update gradParams
function Module:updateGradParameters(momFactor, momDamp, momNesterov)
   -- this allows each module to set its own hyper-parameters
   momFactor = self.momFactor or momFactor
   if momFactor <= 0 then
      return
   end
   momDamp = self.momDamp or momDamp or momFactor
   if self.momNesterov ~= nil then
      momNesterov = self.momNesterov
   end

   if self.modules then
      for i,module in ipairs(self.modules) do
         module:updateGradParameters(momFactor, momDamp, momNesterov)
      end
   else
      local params, gradParams = self:parameters()
      if (not params) or _.isEmpty(params) then
         return
      end
      local momGradParams = self:momentumGradParameters()
      for i,gradParam in pairs(gradParams) do
         momGradParams[i]:mul(momFactor)
         momGradParams[i]:add(1-momDamp, gradParam)
      end

      if momNesterov then
         for i,gradParam in pairs(gradParams) do
            gradParam:add(momFactor, momGradParams[i])
         end
      else
         for i,gradParam in pairs(gradParams) do
            gradParam:copy(momGradParams[i])
         end
      end
   end
end

function Module:checkParameters()
   local params = self:parameters() or {}
   for k,param in pairs(params) do
      if _.isNaN(param:sum()) then
         error("NaN Error for param at index" ..k)
      end
   end
end

function Module:dontBackward()
   self.backward = function() end
   self.updateGradInput = function() end
   self.accGradParameters = function() end
   self.accUpdateGradParameters = function() end
   return self
end

function Module:contiguousInput(input, backward)
   if backward then
      return self.dpnn_cinput or input
   end
   if not input:isContiguous() then
      self.dpnn_cinput = self.dpnn_cinput or input.new()
      self.dpnn_cinput:resizeAs(input):copy(input)
      input = self.dpnn_cinput
   end
   return input
end

function Module:toBatch(tensor, nDim, batchDim)
   local batchDim = batchDim or 1
   if tensor:dim() == nDim then
      self.dpnn_online = true
      local size = tensor:size():totable()
      table.insert(size, batchDim, 1)
      tensor = tensor:view(table.unpack(size))
   else
      self.dpnn_online = false
   end
   return tensor
end

function Module:fromBatch(tensor, batchDim)
   if self.dpnn_online then
      local size = tensor:size():totable()
      assert(table.remove(size, batchDim) == 1)
      tensor = tensor:view(table.unpack(size))
   end
   return tensor
end

function Module:extrapolateType()
   local params = module:parameters()
   if params then
      -- extrapolate the tensor type of the module
      local types = {}
      for i, param in ipairs(params) do
         local tensorType = torch.type(param)
         types[tensorType] = (types[tensorType] or 0) + 1
      end
      local maxCount = 0
      local maxType
      for tensorType, count in pairs(types) do
         if count > maxCount then
            maxtype = tensorType
            maxCount = count
         end
      end
      return maxType
   end
   return nil --unknown otherwise
end

function Module:profile()
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:profile()
      end
   end
   self.dpnn_profile = true
end

function Module:reinforce(reward)
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:reinforce(reward)
      end
   end
end
