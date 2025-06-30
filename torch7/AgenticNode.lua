--[[
Agentic Node Module for torch-central
=====================================

This module implements individual agentic nodes that form the distributed
cognitive architecture. Each agent contains a cognitive grammar kernel,
memory subsystem, task management, and autonomy capabilities.

Key Features:
- Cognitive grammar kernel integration
- Distributed memory management
- Task decomposition and execution
- Attention allocation control
- Meta-cognitive self-modification

Author: torch-central cognitive architecture integration
--]]

local AgenticNode = torch.class('torch.AgenticNode')

function AgenticNode:__init(config)
   config = config or {}
   
   -- Node identification and basic config
   self.nodeId = config.nodeId or 'agent_' .. torch.random(1, 999999)
   self.nodeType = config.nodeType or 'general'
   self.tensorDimensions = config.tensorDimensions or {64, 32}
   
   -- Initialize cognitive grammar kernel
   self.cognitiveKernel = torch.CognitiveGrammar({
      tensorDimensions = self.tensorDimensions,
      memorySize = config.memorySize or 512,
      nodeId = self.nodeId
   })
   
   -- Task management system
   self.taskQueue = {}
   self.currentTask = nil
   self.completedTasks = {}
   
   -- Attention allocation system
   self.attentionWeights = torch.Tensor(10):fill(0.1) -- 10 attention channels
   self.attentionFocus = 1 -- Currently focused channel
   
   -- Autonomy and meta-cognitive state
   self.autonomyLevel = config.autonomyLevel or 0.5
   self.selfModificationEnabled = config.selfModificationEnabled or false
   self.metaStateHistory = {}
   
   -- Performance tracking
   self.performance = {
      tasksCompleted = 0,
      totalActivations = 0,
      averageProcessingTime = 0,
      errorCount = 0
   }
   
   -- Network connections to other agents
   self.agentConnections = {}
   
   self:_initializeDefaultBehaviors()
end

-- Initialize default agent behaviors and task patterns
function AgenticNode:_initializeDefaultBehaviors()
   -- Define basic task types this agent can handle
   self.taskHandlers = {
      ['parse'] = function(self, data) return self:_parseTask(data) end,
      ['learn'] = function(self, data) return self:_learnTask(data) end,
      ['reason'] = function(self, data) return self:_reasonTask(data) end,
      ['modify'] = function(self, data) return self:_modifyTask(data) end
   }
   
   -- Initialize attention patterns
   self.attentionPatterns = {
      ['focus'] = function(channel) return torch.zeros(10):scatter(1, torch.LongTensor{channel}, 1) end,
      ['distribute'] = function() return torch.ones(10):div(10) end,
      ['gradient'] = function(center) 
         local weights = torch.zeros(10)
         for i = 1, 10 do
            weights[i] = math.exp(-math.abs(i - center) / 2)
         end
         return weights:div(weights:sum())
      end
   }
end

-- Add a task to the agent's queue
function AgenticNode:addTask(taskType, taskData, priority)
   priority = priority or 1
   
   local task = {
      id = 'task_' .. torch.random(1, 999999),
      type = taskType,
      data = taskData,
      priority = priority,
      timestamp = os.time(),
      status = 'queued'
   }
   
   table.insert(self.taskQueue, task)
   
   -- Sort by priority (higher priority first)
   table.sort(self.taskQueue, function(a, b) return a.priority > b.priority end)
   
   return task.id
end

-- Process the next task in the queue
function AgenticNode:processNextTask()
   if #self.taskQueue == 0 then
      return nil, "No tasks in queue"
   end
   
   local task = table.remove(self.taskQueue, 1)
   self.currentTask = task
   task.status = 'processing'
   
   local startTime = os.clock()
   local result, error
   
   -- Execute task using appropriate handler
   if self.taskHandlers[task.type] then
      local success, taskResult = pcall(self.taskHandlers[task.type], self, task.data)
      if success then
         result = taskResult
         task.status = 'completed'
      else
         error = taskResult
         task.status = 'error'
         self.performance.errorCount = self.performance.errorCount + 1
      end
   else
      error = "Unknown task type: " .. tostring(task.type)
      task.status = 'error'
      self.performance.errorCount = self.performance.errorCount + 1
   end
   
   -- Update performance metrics
   local processingTime = os.clock() - startTime
   self.performance.totalActivations = self.performance.totalActivations + 1
   if task.status == 'completed' then
      self.performance.tasksCompleted = self.performance.tasksCompleted + 1
      self.performance.averageProcessingTime = 
         (self.performance.averageProcessingTime * (self.performance.tasksCompleted - 1) + processingTime) / 
         self.performance.tasksCompleted
   end
   
   -- Move to completed tasks
   task.result = result
   task.error = error
   task.processingTime = processingTime
   table.insert(self.completedTasks, task)
   self.currentTask = nil
   
   return result, error
end

-- Set attention focus pattern
function AgenticNode:setAttentionPattern(patternName, parameter)
   if not self.attentionPatterns[patternName] then
      error('Unknown attention pattern: ' .. tostring(patternName))
   end
   
   self.attentionWeights = self.attentionPatterns[patternName](parameter or self.attentionFocus)
   return self.attentionWeights
end

-- Connect to another agentic node
function AgenticNode:connectToAgent(otherAgent, connectionType, weight)
   connectionType = connectionType or 'bidirectional'
   weight = weight or 1.0
   
   if torch.typename(otherAgent) ~= 'torch.AgenticNode' then
      error('Can only connect to other AgenticNode instances')
   end
   
   -- Add connection to this agent
   table.insert(self.agentConnections, {
      agent = otherAgent,
      type = connectionType,
      weight = weight,
      messagesSent = 0,
      messagesReceived = 0
   })
   
   -- Also connect cognitive kernels
   self.cognitiveKernel:connectToNode(otherAgent.cognitiveKernel, weight)
   
   -- If bidirectional, add reverse connection
   if connectionType == 'bidirectional' then
      table.insert(otherAgent.agentConnections, {
         agent = self,
         type = connectionType,
         weight = weight,
         messagesSent = 0,
         messagesReceived = 0
      })
      otherAgent.cognitiveKernel:connectToNode(self.cognitiveKernel, weight)
   end
   
   return #self.agentConnections
end

-- Send message to connected agents
function AgenticNode:sendMessage(message, targetAgentId)
   local messagesSent = 0
   
   for _, connection in ipairs(self.agentConnections) do
      local shouldSend = not targetAgentId or connection.agent.nodeId == targetAgentId
      
      if shouldSend then
         -- Process message through cognitive kernel
         local processedMessage = self.cognitiveKernel:process(message, 'activation')
         
         -- Send to target agent
         connection.agent:receiveMessage(processedMessage, self.nodeId)
         connection.messagesSent = connection.messagesSent + 1
         messagesSent = messagesSent + 1
      end
   end
   
   return messagesSent
end

-- Receive message from another agent
function AgenticNode:receiveMessage(message, senderAgentId)
   -- Find sender connection
   for _, connection in ipairs(self.agentConnections) do
      if connection.agent.nodeId == senderAgentId then
         connection.messagesReceived = connection.messagesReceived + 1
         break
      end
   end
   
   -- Process received message through cognitive kernel
   local processedMessage = self.cognitiveKernel:process(message, 'memory_encode')
   
   -- Store in memory if significant
   if processedMessage:norm() > 0.1 then
      local memoryIndex = torch.random(1, self.cognitiveKernel.memorySize)
      self.cognitiveKernel:storeMemory(processedMessage, memoryIndex)
   end
   
   return processedMessage
end

-- Meta-cognitive self-reflection and adaptation
function AgenticNode:performMetaCognition()
   local currentState = {
      performance = self.performance,
      queueLength = #self.taskQueue,
      attentionState = self.attentionWeights:clone(),
      timestamp = os.time()
   }
   
   table.insert(self.metaStateHistory, currentState)
   
   -- Keep only recent history
   if #self.metaStateHistory > 100 then
      table.remove(self.metaStateHistory, 1)
   end
   
   -- Perform optimization based on recent performance
   if self.selfModificationEnabled and #self.metaStateHistory > 10 then
      self:_adaptBehavior()
   end
   
   -- Optimize cognitive kernel
   self.cognitiveKernel:optimize()
   
   return currentState
end

-- Task handler implementations
function AgenticNode:_parseTask(data)
   if torch.isTensor(data) then
      return self.cognitiveKernel:process(data, 'activation')
   else
      error('Parse task requires tensor input')
   end
end

function AgenticNode:_learnTask(data)
   if torch.isTensor(data) then
      local processed = self.cognitiveKernel:process(data, 'memory_encode')
      local memoryIndex = torch.random(1, self.cognitiveKernel.memorySize)
      self.cognitiveKernel:storeMemory(processed, memoryIndex)
      return processed
   else
      error('Learn task requires tensor input')
   end
end

function AgenticNode:_reasonTask(data)
   if torch.isTensor(data) then
      -- Retrieve relevant memory and apply symbolic reasoning
      local memoryIndex = torch.random(1, self.cognitiveKernel.memorySize)
      local memory = self.cognitiveKernel:retrieveMemory(memoryIndex)
      return self.cognitiveKernel:applySymbolicOperation('associate', data, memory)
   else
      error('Reason task requires tensor input')
   end
end

function AgenticNode:_modifyTask(data)
   if self.selfModificationEnabled then
      -- Simple self-modification: adjust attention weights
      if torch.isTensor(data) and data:nElement() == 10 then
         self.attentionWeights = data:clone()
         return data
      else
         error('Modify task requires tensor with 10 elements for attention weights')
      end
   else
      error('Self-modification is disabled for this agent')
   end
end

-- Adaptive behavior modification
function AgenticNode:_adaptBehavior()
   local recentStates = {}
   local historySize = math.min(#self.metaStateHistory, 10)
   
   for i = #self.metaStateHistory - historySize + 1, #self.metaStateHistory do
      table.insert(recentStates, self.metaStateHistory[i])
   end
   
   -- Calculate average error rate
   local totalErrors = 0
   for _, state in ipairs(recentStates) do
      totalErrors = totalErrors + state.performance.errorCount
   end
   local avgErrorRate = totalErrors / historySize
   
   -- Adapt based on error rate
   if avgErrorRate > 0.1 then
      -- High error rate: increase attention focus
      self:setAttentionPattern('focus', self.attentionFocus)
   elseif avgErrorRate < 0.05 then
      -- Low error rate: can distribute attention more
      self:setAttentionPattern('distribute')
   end
end

-- Get comprehensive agent status
function AgenticNode:getStatus()
   return {
      nodeId = self.nodeId,
      nodeType = self.nodeType,
      currentTask = self.currentTask,
      queueLength = #self.taskQueue,
      completedTasks = #self.completedTasks,
      performance = self.performance,
      attentionFocus = self.attentionFocus,
      connectionCount = #self.agentConnections,
      cognitiveKernelInfo = self.cognitiveKernel:getNodeInfo(),
      autonomyLevel = self.autonomyLevel,
      selfModificationEnabled = self.selfModificationEnabled
   }
end

return AgenticNode