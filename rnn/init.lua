require 'torchx'
local _ = require 'moses'
require 'nn'
local _cuda, _ = pcall(require, 'cunn')

-- create global rnn table:
rnn = {}
rnn.cuda = _cuda
rnn.version = 2.7 -- better support for bidirection RNNs

-- lua 5.2 compat

function nn.require(packagename)
   assert(torch.type(packagename) == 'string')
   local success, message = pcall(function() require(packagename) end)
   if not success then
      print("missing package "..packagename..": run 'luarocks install '"..packagename.."'")
      error(message)
   end
end


-- c lib:
require "paths"
pcall(function() paths.require 'librnn' end) -- Not sure why this works...
pcall(function() paths.require 'librnn' end)

unpack = unpack or table.unpack

require('rnn.utils')
-- extensions to existing nn.Module
require('rnn.Module')
require('rnn.Container')
require('rnn.Sequential')
require('rnn.ParallelTable')
require('rnn.LookupTable')
require('rnn.Dropout')
require('rnn.BatchNormalization')

-- extensions to existing nn.Criterion
require('rnn.Criterion')

-- modules
require('rnn.LookupTableMaskZero')
require('rnn.MaskZero')
require('rnn.ReverseSequence')
require('rnn.SpatialGlimpse')
require('rnn.ArgMax')
require('rnn.CategoricalEntropy')
require('rnn.TotalDropout')
require('rnn.SAdd')
require('rnn.CopyGrad')
require('rnn.VariableLength')
require('rnn.StepLSTM')
require('rnn.StepGRU')
require('rnn.ReverseUnreverse')

-- Noise Contrastive Estimation
require('rnn.NCEModule')
require('rnn.NCECriterion')

-- REINFORCE
require('rnn.Reinforce')
require('rnn.ReinforceGamma')
require('rnn.ReinforceBernoulli')
require('rnn.ReinforceNormal')
require('rnn.ReinforceCategorical')

-- REINFORCE criterions
require('rnn.VRClassReward')
require('rnn.BinaryClassReward')

-- for testing:
require('rnn.test')
require('rnn.bigtest')

-- recurrent modules
require('rnn.AbstractRecurrent')
require('rnn.Recursor')
require('rnn.Recurrence')
require('rnn.LinearRNN')
require('rnn.LookupRNN')
require('rnn.RecLSTM')
require('rnn.RecGRU')
require('rnn.GRU')
require('rnn.Mufuru')
require('rnn.NormStabilizer')

-- sequencer modules
require('rnn.AbstractSequencer')
require('rnn.Repeater')
require('rnn.Sequencer')
require('rnn.BiSequencer')
require('rnn.RecurrentAttention')

-- sequencer + recurrent modules
require('rnn.SeqLSTM')
require('rnn.SeqGRU')
require('rnn.SeqBLSTM')
require('rnn.SeqBGRU')

-- recurrent criterions:
require('rnn.AbstractSequencerCriterion')
require('rnn.SequencerCriterion')
require('rnn.RepeaterCriterion')
require('rnn.MaskZeroCriterion')

-- deprecated modules
require('rnn.LSTM')
require('rnn.FastLSTM')
require('rnn.SeqLSTMP')
require('rnn.SeqReverseSequence')
require('rnn.BiSequencerLM')


require('rnn.measure')

-- prevent likely name conflicts
nn.rnn = rnn

return rnn
