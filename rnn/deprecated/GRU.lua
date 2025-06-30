------------------------------------------------------------------------
--[[ GRU ]]--
-- Author: Jin-Hwa Kim
-- License: LICENSE.2nd.txt

-- Gated Recurrent Units architecture.
-- http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
-- Expects 1D or 2D input.
-- The first input in sequence uses zero value for cell and hidden state
--
-- For p > 0, it becomes Bayesian GRUs [Moon et al., 2015; Gal, 2015].
-- In this case, please do not dropout on input as BGRUs handle the input with
-- its own dropouts. First, try 0.25 for p as Gal (2016) suggested, presumably,
-- because of summations of two parts in GRUs connections.
------------------------------------------------------------------------
local GRU, parent = torch.class('nn.GRU', 'nn.AbstractRecurrent')

function GRU:__init(inputSize, outputSize, rho, p, mono)
   self.p = p or 0
   if p and p ~= 0 then
      assert(nn.Dropout(p,false,false,true).lazy, 'only work with Lazy Dropout!')
   end
   self.mono = mono or false
   self.inputSize = inputSize
   self.outputSize = outputSize
   -- build the model
   local stepmodule = self:buildModel()
   parent.__init(self, stepmodule)

   -- for output(0), cell(0) and gradCell(T)
   self.zeroTensor = torch.Tensor()

   self.cells = {}
   self.gradCells = {}
end

-------------------------- factory methods -----------------------------
function GRU:buildModel()
   -- input : {input, prevOutput}
   -- output : {output}

   -- Calculate all four gates in one go : input, hidden, forget, output
   if self.p ~= 0 then
      self.i2g = nn.Sequential()
                     :add(nn.ConcatTable()
                        :add(nn.Dropout(self.p,false,false,true,self.mono))
                        :add(nn.Dropout(self.p,false,false,true,self.mono)))
                     :add(nn.ParallelTable()
                        :add(nn.Linear(self.inputSize, self.outputSize))
                        :add(nn.Linear(self.inputSize, self.outputSize)))
                     :add(nn.JoinTable(2))
      self.o2g = nn.Sequential()
                     :add(nn.ConcatTable()
                        :add(nn.Dropout(self.p,false,false,true,self.mono))
                        :add(nn.Dropout(self.p,false,false,true,self.mono)))
                     :add(nn.ParallelTable()
                        :add(nn.Linear(self.outputSize, self.outputSize):noBias())
                        :add(nn.Linear(self.outputSize, self.outputSize):noBias()))
                     :add(nn.JoinTable(2))
   else
      self.i2g = nn.Linear(self.inputSize, 2*self.outputSize)
      self.o2g = nn.Linear(self.outputSize, 2*self.outputSize):noBias()
   end

   local para = nn.ParallelTable():add(self.i2g):add(self.o2g)
   local gates = nn.Sequential()
   gates:add(para)
   gates:add(nn.CAddTable())

   -- Reshape to (batch_size, n_gates, hid_size)
   -- Then slize the n_gates dimension, i.e dimension 2
   gates:add(nn.Reshape(2,self.outputSize))
   gates:add(nn.SplitTable(1,2))
   local transfer = nn.ParallelTable()
   transfer:add(nn.Sigmoid()):add(nn.Sigmoid())
   gates:add(transfer)

   local concat = nn.ConcatTable():add(nn.Identity()):add(gates)
   local seq = nn.Sequential()
   seq:add(concat)
   seq:add(nn.FlattenTable()) -- x(t), s(t-1), r, z

   -- Rearrange to x(t), s(t-1), r, z, s(t-1)
   local concat = nn.ConcatTable()  --
   concat:add(nn.NarrowTable(1,4)):add(nn.SelectTable(2))
   seq:add(concat):add(nn.FlattenTable())

   -- h
   local hidden = nn.Sequential()
   local concat = nn.ConcatTable()
   local t1 = nn.Sequential()
   t1:add(nn.SelectTable(1))
   local t2 = nn.Sequential()
   t2:add(nn.NarrowTable(2,2)):add(nn.CMulTable())
   if self.p ~= 0 then
      t1:add(nn.Dropout(self.p,false,false,true,self.mono))
      t2:add(nn.Dropout(self.p,false,false,true,self.mono))
   end
   t1:add(nn.Linear(self.inputSize, self.outputSize))
   t2:add(nn.Linear(self.outputSize, self.outputSize):noBias())

   concat:add(t1):add(t2)
   hidden:add(concat):add(nn.CAddTable()):add(nn.Tanh())

   local z1 = nn.Sequential()
   z1:add(nn.SelectTable(4))
   z1:add(nn.SAdd(-1, true))  -- Scalar add & negation

   local z2 = nn.Sequential()
   z2:add(nn.NarrowTable(4,2))
   z2:add(nn.CMulTable())

   local o1 = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(hidden):add(z1)
   o1:add(concat):add(nn.CMulTable())

   local o2 = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(o1):add(z2)
   o2:add(concat):add(nn.CAddTable())

   seq:add(o2)

   return seq
end

------------------------- forward backward -----------------------------
function GRU:_updateOutput(input)
   local prevOutput = self:getHiddenState(self.step-1, input)

   -- output(t) = gru{input(t), output(t-1)}
   local output
   if self.train ~= false then
      local stepmodule = self:getStepModule(self.step)
      -- the actual forward propagation
      output = stepmodule:updateOutput{input, prevOutput}
   else
      output = self.modules[1]:updateOutput{input, prevOutput}
   end

   return output
end

function GRU:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local stepmodule = self:getStepModule(step)

   -- backward propagate through this step
   local _gradOutput = self:getGradHiddenState(step, input)
   assert(_gradOutput)
   self._gradOutputs[step] = nn.utils.recursiveCopy(self._gradOutputs[step], _gradOutput)
   nn.utils.recursiveAdd(self._gradOutputs[step], gradOutput)
   gradOutput = self._gradOutputs[step]

   local gradInputTable = stepmodule:updateGradInput({input, self:getHiddenState(step-1)}, gradOutput)

   self:setGradHiddenState(step-1, gradInputTable[2])

   return gradInputTable[1]
end

function GRU:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)

   -- set the output/gradOutput states of current Module
   local stepmodule = self:getStepModule(step)

   -- backward propagate through this step
   local gradOutput = self._gradOutputs[step] or self:getGradHiddenState(step)
   stepmodule:accGradParameters({input, self:getHiddenState(step-1)}, gradOutput, scale)
end

function GRU:__tostring__()
   return string.format('%s(%d -> %d, %.2f)', torch.type(self), self.inputSize, self.outputSize, self.p)
end

-- migrate GRUs params to BGRUs params
function GRU:migrate(params)
   local _params = self:parameters()
   assert(self.p ~= 0, 'only support for BGRUs.')
   assert(#params == 6, '# of source params should be 6.')
   assert(#_params == 9, '# of destination params should be 9.')
   _params[1]:copy(params[1]:narrow(1,1,self.outputSize))
   _params[2]:copy(params[2]:narrow(1,1,self.outputSize))
   _params[3]:copy(params[1]:narrow(1,self.outputSize+1,self.outputSize))
   _params[4]:copy(params[2]:narrow(1,self.outputSize+1,self.outputSize))
   _params[5]:copy(params[3]:narrow(1,1,self.outputSize))
   _params[6]:copy(params[3]:narrow(1,self.outputSize+1,self.outputSize))
   _params[7]:copy(params[4])
   _params[8]:copy(params[5])
   _params[9]:copy(params[6])
end

function GRU:initZeroTensor(input)
   if input then
      if input:dim() == 2 then
         self.zeroTensor:resize(input:size(1), self.outputSize):zero()
      else
         self.zeroTensor:resize(self.outputSize):zero()
      end
   end
end

function GRU:getHiddenState(step, input)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   local prevOutput
   if step == 0 then
      if self.startState then
         prevOutput = self.startState
      else
         prevOutput = self.zeroTensor
         self:initZeroTensor(input)
      end
   else
      -- previous output and cell of this module
      prevOutput = self.outputs[step]
   end
   return prevOutput
end

function GRU:setHiddenState(step, hiddenState)
   step = step == nil and (self.step - 1) or (step < 0) and (self.step - step - 1) or step
   assert(torch.isTensor(hiddenState))
   if step == 0 then
      self:setStartState(hiddenState)
   else
      self.outputs[step] = hiddenState
   end
end

function GRU:getGradHiddenState(step, input)
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
   local gradOutput
   if step == self.step-1 and not self.gradOutputs[step] then
      if self.startState then
         self:initZeroTensor(input)
      end
      gradOutput = self.zeroTensor
   else
      gradOutput = self.gradOutputs[step]
   end
   return gradOutput
end

function GRU:setGradHiddenState(step, gradHiddenState)
   local _step = self.updateGradInputStep or self.step
   step = step == nil and (_step - 1) or (step < 0) and (_step - step - 1) or step
   assert(torch.isTensor(gradHiddenState))
   self.gradOutputs[step] = gradHiddenState
end

