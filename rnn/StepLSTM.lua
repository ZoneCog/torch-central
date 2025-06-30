-- StepLSTM is a step-wise module that can be used inside Recurrence to implement an LSTM.
-- That is, the StepLSTM efficiently implements a single LSTM time-step.
-- Its efficient because it doesn't use any internal modules; it calls BLAS directly.
-- StepLSTM is based on SeqLSTM.
-- Input : {input[t], hidden[t-1], cell[t-1])}
-- Output: {hidden[t], cell[t]}
local StepLSTM, parent = torch.class('nn.StepLSTM', 'nn.Module')

function StepLSTM:__init(inputsize, hiddensize, outputsize)
   parent.__init(self)
   if hiddensize and outputsize then
      -- implements LSTMP
      self.weightO = torch.Tensor(hiddensize, outputsize)
      self.gradWeightO = torch.Tensor(hiddensize, outputsize)
   else
      -- implements LSTM
      assert(inputsize and hiddensize and not outputsize)
      outputsize = hiddensize
   end
   self.inputsize, self.hiddensize, self.outputsize = inputsize, hiddensize, outputsize

   self.weight = torch.Tensor(inputsize+outputsize, 4 * hiddensize)
   self.gradWeight = torch.Tensor(inputsize+outputsize, 4 * hiddensize)

   self.bias = torch.Tensor(4 * hiddensize)
   self.gradBias = torch.Tensor(4 * hiddensize):zero()
   self:reset()

   self.gates = torch.Tensor() -- batchsize x 4*outputsize

   self.output = {torch.Tensor(), torch.Tensor()}
   self.gradInput = {torch.Tensor(), torch.Tensor(), torch.Tensor()}

   -- set this to true for variable length sequences that seperate
   -- independent sequences with a step of zeros (a tensor of size D)
   self.maskzero = false
   self.v2 = true
end

function StepLSTM:reset(std)
   self.bias:zero()
   self.bias[{{self.outputsize + 1, 2 * self.outputsize}}]:fill(1)
   self.weight:normal(0, std or (1.0 / math.sqrt(self.hiddensize + self.inputsize)))
   if self.weightO then
      self.weightO:normal(0, std or (1.0 / math.sqrt(self.outputsize + self.hiddensize)))
   end
   return self
end

function StepLSTM:updateOutput(input)
   self.recompute_backward = true
   local cur_x, prev_h, prev_c = input[1], input[2], input[3]
   local next_h, next_c = self.output[1], self.output[2]
   if cur_x.nn and cur_x.nn.StepLSTM_updateOutput and not self.forceLua then
      if self.weightO then -- LSTMP
         self.hidden = self.hidden or cur_x.new()
         cur_x.nn.StepLSTM_updateOutput(self.weight, self.bias, self.gates,
                                        cur_x, prev_h, prev_c,
                                        self.inputsize, self.hiddensize, self.outputsize,
                                        self.hidden, next_c, self.weightO, next_h)
      else -- LSTM
         cur_x.nn.StepLSTM_updateOutput(self.weight, self.bias, self.gates,
                                        cur_x, prev_h, prev_c,
                                        self.inputsize, self.hiddensize, self.outputsize,
                                        next_h, next_c)
      end
   else
      if self.weightO then -- LSTMP
         self.hidden = self.hidden or cur_x.new()
         next_h = self.hidden
      end
      assert(torch.isTensor(prev_h))
      assert(torch.isTensor(prev_c))
      local batchsize, inputsize, hiddensize = cur_x:size(1), cur_x:size(2), self.hiddensize
      assert(inputsize == self.inputsize)

      -- TODO use self.bias_view
      local bias_expand = self.bias:view(1, 4 * hiddensize):expand(batchsize, 4 * hiddensize)
      local Wx = self.weight:narrow(1,1,inputsize)
      local Wh = self.weight:narrow(1,inputsize+1,self.outputsize)

      next_h:resize(batchsize, hiddensize)
      next_c:resize(batchsize, hiddensize)

      local gates = self.gates
      local nElement = gates:nElement()
      gates:resize(batchsize, 4 * hiddensize)
      if gates:nElement() ~= batchsize * 4 * hiddensize then
         gates:zero()
      end

      -- forward
      gates:addmm(bias_expand, cur_x, Wx)
      gates:addmm(prev_h, Wh)
      gates[{{}, {1, 3 * hiddensize}}]:sigmoid()
      gates[{{}, {3 * hiddensize + 1, 4 * hiddensize}}]:tanh()
      local input_gate = gates[{{}, {1, hiddensize}}]
      local forget_gate = gates[{{}, {hiddensize + 1, 2 * hiddensize}}]
      local output_gate = gates[{{}, {2 * hiddensize + 1, 3 * hiddensize}}]
      local input_transform = gates[{{}, {3 * hiddensize + 1, 4 * hiddensize}}]
      next_h:cmul(input_gate, input_transform)
      next_c:cmul(forget_gate, prev_c):add(next_h)
      next_h:tanh(next_c):cmul(output_gate)

      if self.weightO then -- LSTMP
         self.output[1]:resize(batchsize, self.outputsize)
         self.output[1]:mm(next_h, self.weightO)
      end
   end

   if self.maskzero and self.zeroMask ~= false then
      if self.v2 then
         assert(self.zeroMask ~= nil, torch.type(self).." expecting zeroMask tensor or false")
      else -- backwards compat
         self.zeroMask = nn.utils.getZeroMaskBatch(cur_x, self.zeroMask)
      end
      -- zero masked outputs and gates
      nn.utils.recursiveZeroMask({next_h, next_c, self.gates}, self.zeroMask)
   end

   return self.output
end

function StepLSTM:backward(input, gradOutput, scale)
   self.recompute_backward = false
   local cur_x, prev_h, prev_c = input[1], input[2], input[3]
   local grad_next_h, grad_next_c = gradOutput[1], gradOutput[2]
   local next_c = self.output[2]
   local grad_cur_x, grad_prev_h, grad_prev_c = self.gradInput[1], self.gradInput[2], self.gradInput[3]
   scale = scale or 1.0
   assert(scale == 1.0, 'must have scale=1')

   local grad_gates = torch.getBuffer('StepLSTM', 'grad_gates', self.gates) -- batchsize x 4*outputsize
   local grad_gates_sum = torch.getBuffer('StepLSTM', 'grad_gates_sum', self.gates) -- 1 x 4*outputsize

   if self.maskzero and self.zeroMask ~= false then
      -- zero masked gradOutput
      nn.utils.recursiveZeroMask({grad_next_h, grad_next_c}, self.zeroMask)
   end

   if cur_x.nn and cur_x.nn.StepLSTM_backward and not self.forceLua then
      if self.weightO then -- LSTMP
         local grad_hidden = torch.getBuffer('StepLSTM', 'grad_hidden', self.hidden)
         cur_x.nn.StepLSTM_backward(self.weight, self.gates,
                                    self.gradWeight, self.gradBias, grad_gates, grad_gates_sum,
                                    cur_x, prev_h, prev_c, next_c, grad_next_h, grad_next_c,
                                    scale, self.inputsize, self.hiddensize, self.outputsize,
                                    grad_cur_x, grad_prev_h, grad_prev_c,
                                    self.weightO, self.hidden, self.gradWeightO, grad_hidden)
      else -- LSTM
         cur_x.nn.StepLSTM_backward(self.weight, self.gates,
                                    self.gradWeight, self.gradBias, grad_gates, grad_gates_sum,
                                    cur_x, prev_h, prev_c, next_c, grad_next_h, grad_next_c,
                                    scale, self.inputsize, self.hiddensize, self.outputsize,
                                    grad_cur_x, grad_prev_h, grad_prev_c)
      end
   else
      local batchsize, inputsize, hiddensize = cur_x:size(1), cur_x:size(2), self.hiddensize
      assert(inputsize == self.inputsize)

      if self.weightO then -- LSTMP
         local grad_hidden = torch.getBuffer('StepLSTM', 'grad_hidden', self.hidden)

         self.gradWeightO:addmm(scale, self.hidden:t(), grad_next_h)
         grad_hidden:resize(batchsize, hiddensize)
         grad_hidden:mm(grad_next_h, self.weightO:t())
         grad_next_h = grad_hidden
      end

      grad_cur_x:resize(batchsize, inputsize)
      grad_prev_h:resize(batchsize, self.outputsize)
      grad_prev_c:resize(batchsize, hiddensize)

      local Wx = self.weight:narrow(1,1,inputsize)
      local Wh = self.weight:narrow(1,inputsize+1,self.outputsize)
      local grad_Wx = self.gradWeight:narrow(1,1,inputsize)
      local grad_Wh = self.gradWeight:narrow(1,inputsize+1,self.outputsize)
      local grad_b = self.gradBias

      local gates = self.gates

      -- backward

      local input_gate = gates[{{}, {1, hiddensize}}]
      local forget_gate = gates[{{}, {hiddensize + 1, 2 * hiddensize}}]
      local output_gate = gates[{{}, {2 * hiddensize + 1, 3 * hiddensize}}]
      local input_transform = gates[{{}, {3 * hiddensize + 1, 4 * hiddensize}}]

      grad_gates:resize(batchsize, 4 * hiddensize)

      local grad_input_gate = grad_gates[{{}, {1, hiddensize}}]
      local grad_forget_gate = grad_gates[{{}, {hiddensize + 1, 2 * hiddensize}}]
      local grad_output_gate = grad_gates[{{}, {2 * hiddensize + 1, 3 * hiddensize}}]
      local grad_input_transform = grad_gates[{{}, {3 * hiddensize + 1, 4 * hiddensize}}]

      -- we use grad_[input,forget,output]_gate as temporary buffers to compute grad_prev_c.
      grad_input_gate:tanh(next_c)
      grad_forget_gate:cmul(grad_input_gate, grad_input_gate)
      grad_output_gate:fill(1):add(-1, grad_forget_gate):cmul(output_gate):cmul(grad_next_h)
      grad_prev_c:add(grad_next_c, grad_output_gate)

      -- we use above grad_input_gate to compute grad_output_gate
      grad_output_gate:fill(1):add(-1, output_gate):cmul(output_gate):cmul(grad_input_gate):cmul(grad_next_h)

      -- Use grad_input_gate as a temporary buffer for computing grad_input_transform
      grad_input_gate:cmul(input_transform, input_transform)
      grad_input_transform:fill(1):add(-1, grad_input_gate):cmul(input_gate):cmul(grad_prev_c)

      -- We don't need any temporary storage for these so do them last
      grad_input_gate:fill(1):add(-1, input_gate):cmul(input_gate):cmul(input_transform):cmul(grad_prev_c)
      grad_forget_gate:fill(1):add(-1, forget_gate):cmul(forget_gate):cmul(prev_c):cmul(grad_prev_c)

      grad_cur_x:mm(grad_gates, Wx:t())
      grad_Wx:addmm(scale, cur_x:t(), grad_gates)
      grad_Wh:addmm(scale, prev_h:t(), grad_gates)
      grad_gates_sum:resize(1, 4 * hiddensize):sum(grad_gates, 1)
      grad_b:add(scale, grad_gates_sum)

      grad_prev_h:mm(grad_gates, Wh:t())
      grad_prev_c:cmul(forget_gate)
   end

   return self.gradInput
end

function StepLSTM:updateGradInput(input, gradOutput)
   if self.recompute_backward then
      self:backward(input, gradOutput, 1.0)
   end
   return self.gradInput
end

function StepLSTM:accGradParameters(input, gradOutput, scale)
   if self.recompute_backward then
      self:backward(input, gradOutput, scale)
   end
end

function StepLSTM:clearState()
   self.gates:set()

   self.output[1]:set(); self.output[2]:set()
   self.gradInput[1]:set(); self.gradInput[2]:set(); self.gradInput[3]:set()
end

function StepLSTM:type(type, ...)
   self:clearState()
   return parent.type(self, type, ...)
end

function StepLSTM:parameters()
   return {self.weight, self.bias, self.weightO}, {self.gradWeight, self.gradBias, self.gradWeightO}
end

function StepLSTM:maskZero(v1)
   self.maskzero = true
   self.v2 = not v1
   return self
end

StepLSTM.setZeroMask = nn.MaskZero.setZeroMask

function StepLSTM:__tostring__()
   if self.weightO then
       return self.__typename .. string.format("(%d -> %d -> %d)", self.inputsize, self.hiddensize, self.outputsize)
   else
       return self.__typename .. string.format("(%d -> %d)", self.inputsize, self.outputsize)
   end
end

-- for sharedClone
local _ = require 'moses'
local params = _.clone(parent.dpnn_parameters)
table.insert(params, 'weightO')
StepLSTM.dpnn_parameters = params

local gradParams = _.clone(parent.dpnn_gradParameters)
table.insert(gradParams, 'gradWeightO')
StepLSTM.dpnn_gradParameters = gradParams
