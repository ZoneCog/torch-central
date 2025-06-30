------------------------------------------------------------------------
--[[ Recursor ]]--
-- Decorates module to be used within an AbstractSequencer.
-- It does this by making the decorated module conform to the
-- AbstractRecurrent interface (which is inherited by LSTM/Recurrent)
------------------------------------------------------------------------
local Recursor, parent = torch.class('nn.Recursor', 'nn.AbstractRecurrent')

function Recursor:_updateOutput(input)
   local output
   if self.train ~= false then -- if self.train or self.train == nil then
      local stepmodule = self:getStepModule(self.step)
      output = stepmodule:updateOutput(input)
   else
      output = self.modules[1]:updateOutput(input)
   end

   return output
end

function Recursor:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)

   local stepmodule = self:getStepModule(step)
   stepmodule:setOutputStep(step)
   local gradInput = stepmodule:updateGradInput(input, gradOutput)

   return gradInput
end

function Recursor:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)

   local stepmodule = self:getStepModule(step)
   stepmodule:setOutputStep(step)
   stepmodule:accGradParameters(input, gradOutput, scale)
end

function Recursor:includingSharedClones(f)
   local modules = self.modules
   self.modules = {}
   local sharedClones = self.sharedClones
   self.sharedClones = nil
   for i,modules in ipairs{modules, sharedClones} do
      for j, module in pairs(modules) do
         table.insert(self.modules, module)
      end
   end
   local r = {f()}
   self.modules = modules
   self.sharedClones = sharedClones
   return unpack(r)
end

function Recursor:forget(offset)
   parent.forget(self, offset)
   nn.Module.forget(self)
   return self
end

function Recursor:maxBPTTstep(seqlen)
   self.seqlen = seqlen
   nn.Module.maxBPTTstep(self, seqlen)
end

function Recursor:getHiddenState(...)
   return self.modules[1]:getHiddenState(...)
end

function Recursor:setHiddenState(...)
   return self.modules[1]:setHiddenState(...)
end

function Recursor:getGradHiddenState(...)
   return self.modules[1]:getGradHiddenState(...)
end

function Recursor:setGradHiddenState(...)
   return self.modules[1]:setGradHiddenState(...)
end

function Recursor:setStartState(...)
   return self.modules[1]:setStartState(...)
end

Recursor.__tostring__ = nn.Decorator.__tostring__
