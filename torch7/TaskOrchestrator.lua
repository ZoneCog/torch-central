--[[
Task Orchestrator Module for torch-central
==========================================

This module implements the task orchestration layer for the distributed
cognitive architecture. It manages task decomposition, attention allocation,
and coordination between agentic nodes.

Key Features:
- Task decomposition and planning
- Attention allocation control (ECAN-inspired)
- Inter-agent task coordination
- Cognitive resource management
- Adaptive task scheduling

Author: torch-central cognitive architecture integration
--]]

local TaskOrchestrator = torch.class('torch.TaskOrchestrator')

function TaskOrchestrator:__init(config)
   config = config or {}
   
   -- Orchestrator configuration
   self.maxConcurrentTasks = config.maxConcurrentTasks or 10
   self.maxAgents = config.maxAgents or 50
   self.attentionBudget = config.attentionBudget or 100.0
   
   -- Agent management
   self.registeredAgents = {}
   self.agentCapabilities = {}
   self.agentWorkloads = {}
   
   -- Task management
   self.taskQueue = {}
   self.activeTasks = {}
   self.completedTasks = {}
   self.taskIdCounter = 1
   
   -- Attention allocation system (ECAN-inspired)
   self.attentionMap = {}
   self.importanceValues = {}
   self.urgencyValues = {}
   self.confidenceValues = {}
   
   -- Resource management
   self.resourcePools = {
      computational = config.computationalBudget or 1000.0,
      memory = config.memoryBudget or 2048,
      bandwidth = config.bandwidthBudget or 100.0
   }
   self.resourceUsage = {
      computational = 0,
      memory = 0,
      bandwidth = 0
   }
   
   -- Performance metrics
   self.metrics = {
      tasksCreated = 0,
      tasksCompleted = 0,
      tasksDecomposed = 0,
      agentAssignments = 0,
      attentionAllocations = 0,
      resourceOptimizations = 0
   }
   
   -- Task decomposition patterns
   self.decompositionPatterns = {}
   self:_initializeDecompositionPatterns()
   
   -- Scheduling algorithms
   self.schedulingMode = config.schedulingMode or 'priority'
   self.schedulers = {
      priority = function(tasks) return self:_priorityScheduler(tasks) end,
      roundrobin = function(tasks) return self:_roundRobinScheduler(tasks) end,
      cognitive = function(tasks) return self:_cognitiveScheduler(tasks) end
   }
end

-- Initialize default task decomposition patterns
function TaskOrchestrator:_initializeDecompositionPatterns()
   self.decompositionPatterns = {
      -- Sequential decomposition
      sequential = function(task, subtaskCount)
         local subtasks = {}
         for i = 1, subtaskCount do
            table.insert(subtasks, {
               id = task.id .. '_seq_' .. i,
               type = task.type,
               data = task.data,
               priority = task.priority,
               dependencies = i > 1 and {task.id .. '_seq_' .. (i-1)} or {},
               parentTask = task.id
            })
         end
         return subtasks
      end,
      
      -- Parallel decomposition
      parallel = function(task, subtaskCount)
         local subtasks = {}
         for i = 1, subtaskCount do
            table.insert(subtasks, {
               id = task.id .. '_par_' .. i,
               type = task.type,
               data = task.data,
               priority = task.priority,
               dependencies = {},
               parentTask = task.id
            })
         end
         return subtasks
      end,
      
      -- Hierarchical decomposition
      hierarchical = function(task, levels)
         local subtasks = {}
         local currentLevel = {task}
         
         for level = 1, levels do
            local nextLevel = {}
            for _, parentTask in ipairs(currentLevel) do
               for i = 1, 2 do -- Binary decomposition
                  local subtask = {
                     id = parentTask.id .. '_hier_' .. level .. '_' .. i,
                     type = parentTask.type,
                     data = parentTask.data,
                     priority = parentTask.priority * 0.8, -- Slightly lower priority
                     dependencies = {},
                     parentTask = parentTask.id,
                     level = level
                  }
                  table.insert(subtasks, subtask)
                  table.insert(nextLevel, subtask)
               end
            end
            currentLevel = nextLevel
         end
         
         return subtasks
      end
   }
end

-- Register an agentic node with the orchestrator
function TaskOrchestrator:registerAgent(agent, capabilities)
   if torch.typename(agent) ~= 'torch.AgenticNode' then
      error('Only AgenticNode instances can be registered')
   end
   
   capabilities = capabilities or {'general'}
   
   table.insert(self.registeredAgents, agent)
   self.agentCapabilities[agent.nodeId] = capabilities
   self.agentWorkloads[agent.nodeId] = {
      assignedTasks = 0,
      completedTasks = 0,
      averageProcessingTime = 0,
      utilization = 0
   }
   
   -- Initialize attention allocation for agent
   self.attentionMap[agent.nodeId] = {
      current = 10.0, -- Base attention allocation
      maximum = 50.0,
      minimum = 1.0
   }
   
   return #self.registeredAgents
end

-- Create and queue a new task
function TaskOrchestrator:createTask(taskType, taskData, priority, metadata)
   priority = priority or 1
   metadata = metadata or {}
   
   local task = {
      id = 'task_' .. self.taskIdCounter,
      type = taskType,
      data = taskData,
      priority = priority,
      status = 'queued',
      creationTime = os.time(),
      deadline = metadata.deadline,
      requiredCapabilities = metadata.requiredCapabilities or {'general'},
      estimatedComplexity = metadata.estimatedComplexity or 1,
      dependencies = metadata.dependencies or {},
      parentTask = metadata.parentTask,
      metadata = metadata
   }
   
   self.taskIdCounter = self.taskIdCounter + 1
   table.insert(self.taskQueue, task)
   
   -- Calculate importance, urgency, and confidence values
   self:_calculateTaskValues(task)
   
   -- Update metrics
   self.metrics.tasksCreated = self.metrics.tasksCreated + 1
   
   return task.id
end

-- Decompose a complex task into subtasks
function TaskOrchestrator:decomposeTask(taskId, decompositionType, parameters)
   decompositionType = decompositionType or 'sequential'
   parameters = parameters or {}
   
   local task = self:_findTask(taskId)
   if not task then
      error('Task not found: ' .. tostring(taskId))
   end
   
   if not self.decompositionPatterns[decompositionType] then
      error('Unknown decomposition type: ' .. tostring(decompositionType))
   end
   
   -- Apply decomposition pattern
   local subtasks = self.decompositionPatterns[decompositionType](task, 
      parameters.count or 3, parameters.levels or 2)
   
   -- Add subtasks to queue
   for _, subtask in ipairs(subtasks) do
      table.insert(self.taskQueue, subtask)
      self:_calculateTaskValues(subtask)
   end
   
   -- Mark original task as decomposed
   task.status = 'decomposed'
   task.subtasks = {}
   for _, subtask in ipairs(subtasks) do
      table.insert(task.subtasks, subtask.id)
   end
   
   -- Update metrics
   self.metrics.tasksDecomposed = self.metrics.tasksDecomposed + 1
   
   return subtasks
end

-- Allocate attention to agents and tasks using ECAN-inspired mechanism
function TaskOrchestrator:allocateAttention()
   local totalImportance = 0
   local totalUrgency = 0
   
   -- Calculate total importance and urgency
   for taskId, importance in pairs(self.importanceValues) do
      totalImportance = totalImportance + importance
   end
   for taskId, urgency in pairs(self.urgencyValues) do
      totalUrgency = totalUrgency + urgency
   end
   
   -- Allocate attention budget based on task values
   local attentionAllocations = {}
   
   for taskId, importance in pairs(self.importanceValues) do
      local urgency = self.urgencyValues[taskId] or 0
      local confidence = self.confidenceValues[taskId] or 0.5
      
      -- ECAN-style attention calculation
      local attentionValue = (importance / totalImportance + urgency / totalUrgency) * 
                           confidence * self.attentionBudget / 2
      
      attentionAllocations[taskId] = attentionValue
   end
   
   -- Distribute attention to responsible agents
   for _, agent in ipairs(self.registeredAgents) do
      local agentAttention = self.attentionMap[agent.nodeId].current
      
      -- Find tasks assigned to this agent
      for _, task in ipairs(self.activeTasks) do
         if task.assignedAgent == agent.nodeId then
            local taskAttention = attentionAllocations[task.id] or 0
            agentAttention = agentAttention + taskAttention
         end
      end
      
      -- Update agent attention allocation
      self.attentionMap[agent.nodeId].current = math.min(agentAttention, 
         self.attentionMap[agent.nodeId].maximum)
   end
   
   -- Update metrics
   self.metrics.attentionAllocations = self.metrics.attentionAllocations + 1
   
   return attentionAllocations
end

-- Process task queue and assign tasks to agents
function TaskOrchestrator:processTaskQueue()
   -- Apply current scheduling algorithm
   local scheduler = self.schedulers[self.schedulingMode]
   if not scheduler then
      error('Unknown scheduling mode: ' .. tostring(self.schedulingMode))
   end
   
   local scheduledTasks = scheduler(self.taskQueue)
   local assignmentCount = 0
   
   for _, task in ipairs(scheduledTasks) do
      if #self.activeTasks < self.maxConcurrentTasks then
         local assignedAgent = self:_assignTaskToAgent(task)
         
         if assignedAgent then
            -- Move task from queue to active
           self:_removeFromQueue(task.id)
            table.insert(self.activeTasks, task)
            task.status = 'active'
            task.assignedAgent = assignedAgent.nodeId
            task.startTime = os.time()
            
            -- Assign task to agent
            assignedAgent:addTask(task.type, task.data, task.priority)
            
            -- Update agent workload
            self.agentWorkloads[assignedAgent.nodeId].assignedTasks = 
               self.agentWorkloads[assignedAgent.nodeId].assignedTasks + 1
            
            assignmentCount = assignmentCount + 1
         end
      else
         break -- Cannot assign more tasks
      end
   end
   
   -- Update metrics
   self.metrics.agentAssignments = self.metrics.agentAssignments + assignmentCount
   
   return assignmentCount
end

-- Monitor active tasks and update their status
function TaskOrchestrator:monitorActiveTasks()
   local completedCount = 0
   local tasksToRemove = {}
   
   for i, task in ipairs(self.activeTasks) do
      -- Check if task is completed (simplified check)
      local assignedAgent = self:_findAgent(task.assignedAgent)
      
      if assignedAgent then
         local agentStatus = assignedAgent:getStatus()
         
         -- Check if agent has completed the task
         if agentStatus.completedTasks > self.agentWorkloads[task.assignedAgent].completedTasks then
            -- Task completed
            task.status = 'completed'
            task.completionTime = os.time()
            task.processingTime = task.completionTime - task.startTime
            
            -- Move to completed tasks
            table.insert(self.completedTasks, task)
            table.insert(tasksToRemove, i)
            
            -- Update agent workload
            self.agentWorkloads[task.assignedAgent].completedTasks = agentStatus.completedTasks
            self.agentWorkloads[task.assignedAgent].utilization = 
               agentStatus.performance.tasksCompleted / (agentStatus.performance.tasksCompleted + #agentStatus.taskQueue + 1)
            
            completedCount = completedCount + 1
         end
      end
   end
   
   -- Remove completed tasks from active list
   for i = #tasksToRemove, 1, -1 do
      table.remove(self.activeTasks, tasksToRemove[i])
   end
   
   -- Update metrics
   self.metrics.tasksCompleted = self.metrics.tasksCompleted + completedCount
   
   return completedCount
end

-- Optimize resource usage across the system
function TaskOrchestrator:optimizeResources()
   -- Reset resource usage counters
   self.resourceUsage = {
      computational = 0,
      memory = 0,
      bandwidth = 0
   }
   
   -- Calculate current resource usage
   for _, agent in ipairs(self.registeredAgents) do
      local agentStatus = agent:getStatus()
      
      -- Estimate resource usage based on agent status
      self.resourceUsage.computational = self.resourceUsage.computational + 
         agentStatus.queueLength * 10 + agentStatus.completedTasks * 0.1
      self.resourceUsage.memory = self.resourceUsage.memory + 
         agentStatus.cognitiveKernelInfo.memorySize * 0.1
      self.resourceUsage.bandwidth = self.resourceUsage.bandwidth + 
         agentStatus.connectionCount * 5
   end
   
   -- Apply resource optimization strategies
   local optimizations = 0
   
   -- Computational optimization
   if self.resourceUsage.computational > self.resourcePools.computational * 0.8 then
      self:_optimizeComputationalLoad()
      optimizations = optimizations + 1
   end
   
   -- Memory optimization
   if self.resourceUsage.memory > self.resourcePools.memory * 0.8 then
      self:_optimizeMemoryUsage()
      optimizations = optimizations + 1
   end
   
   -- Bandwidth optimization
   if self.resourceUsage.bandwidth > self.resourcePools.bandwidth * 0.8 then
      self:_optimizeBandwidthUsage()
      optimizations = optimizations + 1
   end
   
   -- Update metrics
   self.metrics.resourceOptimizations = self.metrics.resourceOptimizations + optimizations
   
   return self.resourceUsage
end

-- Get orchestrator status
function TaskOrchestrator:getStatus()
   local queuedTasks = #self.taskQueue
   local activeTasks = #self.activeTasks
   local totalAgents = #self.registeredAgents
   
   -- Calculate average agent utilization
   local totalUtilization = 0
   for _, workload in pairs(self.agentWorkloads) do
      totalUtilization = totalUtilization + workload.utilization
   end
   local avgUtilization = totalAgents > 0 and totalUtilization / totalAgents or 0
   
   return {
      queuedTasks = queuedTasks,
      activeTasks = activeTasks,
      completedTasks = #self.completedTasks,
      registeredAgents = totalAgents,
      averageAgentUtilization = avgUtilization,
      attentionBudget = self.attentionBudget,
      resourceUsage = self.resourceUsage,
      resourcePools = self.resourcePools,
      schedulingMode = self.schedulingMode,
      metrics = self.metrics
   }
end

-- Helper functions

function TaskOrchestrator:_calculateTaskValues(task)
   -- Calculate importance based on priority and complexity
   local importance = task.priority * (task.estimatedComplexity or 1)
   self.importanceValues[task.id] = importance
   
   -- Calculate urgency based on deadline
   local urgency = 1
   if task.deadline then
      local timeLeft = task.deadline - os.time()
      urgency = math.max(0.1, 1 / math.max(timeLeft, 1))
   end
   self.urgencyValues[task.id] = urgency
   
   -- Calculate confidence based on agent capabilities
   local confidence = 0.5 -- Default confidence
   local requiredCaps = task.requiredCapabilities or {'general'}
   
   for _, agent in ipairs(self.registeredAgents) do
      local agentCaps = self.agentCapabilities[agent.nodeId] or {}
      local capMatch = 0
      
      for _, reqCap in ipairs(requiredCaps) do
         for _, agentCap in ipairs(agentCaps) do
            if reqCap == agentCap then
               capMatch = capMatch + 1
               break
            end
         end
      end
      
      local agentConfidence = capMatch / #requiredCaps
      confidence = math.max(confidence, agentConfidence)
   end
   
   self.confidenceValues[task.id] = confidence
end

function TaskOrchestrator:_findTask(taskId)
   -- Search in task queue
   for _, task in ipairs(self.taskQueue) do
      if task.id == taskId then
         return task
      end
   end
   
   -- Search in active tasks
   for _, task in ipairs(self.activeTasks) do
      if task.id == taskId then
         return task
      end
   end
   
   -- Search in completed tasks
   for _, task in ipairs(self.completedTasks) do
      if task.id == taskId then
         return task
      end
   end
   
   return nil
end

function TaskOrchestrator:_findAgent(agentId)
   for _, agent in ipairs(self.registeredAgents) do
      if agent.nodeId == agentId then
         return agent
      end
   end
   return nil
end

function TaskOrchestrator:_removeFromQueue(taskId)
   for i, task in ipairs(self.taskQueue) do
      if task.id == taskId then
         table.remove(self.taskQueue, i)
         break
      end
   end
end

function TaskOrchestrator:_assignTaskToAgent(task)
   local bestAgent = nil
   local bestScore = -1
   
   for _, agent in ipairs(self.registeredAgents) do
      local agentCaps = self.agentCapabilities[agent.nodeId] or {}
      local agentWorkload = self.agentWorkloads[agent.nodeId]
      
      -- Calculate assignment score
      local capabilityScore = 0
      for _, reqCap in ipairs(task.requiredCapabilities or {'general'}) do
         for _, agentCap in ipairs(agentCaps) do
            if reqCap == agentCap then
               capabilityScore = capabilityScore + 1
               break
            end
         end
      end
      
      local utilizationScore = 1 - agentWorkload.utilization
      local totalScore = capabilityScore * 0.7 + utilizationScore * 0.3
      
      if totalScore > bestScore then
         bestScore = totalScore
         bestAgent = agent
      end
   end
   
   return bestAgent
end

-- Scheduler implementations
function TaskOrchestrator:_priorityScheduler(tasks)
   local sortedTasks = {}
   for _, task in ipairs(tasks) do
      table.insert(sortedTasks, task)
   end
   
   table.sort(sortedTasks, function(a, b) return a.priority > b.priority end)
   return sortedTasks
end

function TaskOrchestrator:_roundRobinScheduler(tasks)
   -- Simple round-robin (return tasks in order)
   return tasks
end

function TaskOrchestrator:_cognitiveScheduler(tasks)
   local scoredTasks = {}
   
   for _, task in ipairs(tasks) do
      local importance = self.importanceValues[task.id] or 1
      local urgency = self.urgencyValues[task.id] or 1
      local confidence = self.confidenceValues[task.id] or 0.5
      
      local cognitiveScore = importance * urgency * confidence
      
      table.insert(scoredTasks, {task = task, score = cognitiveScore})
   end
   
   table.sort(scoredTasks, function(a, b) return a.score > b.score end)
   
   local sortedTasks = {}
   for _, scoredTask in ipairs(scoredTasks) do
      table.insert(sortedTasks, scoredTask.task)
   end
   
   return sortedTasks
end

-- Resource optimization functions
function TaskOrchestrator:_optimizeComputationalLoad()
   -- Redistribute tasks from overloaded agents
   for _, agent in ipairs(self.registeredAgents) do
      local agentStatus = agent:getStatus()
      if agentStatus.queueLength > 5 then
         -- Agent is overloaded, try to redistribute some tasks
         -- Implementation would involve moving tasks between agents
      end
   end
end

function TaskOrchestrator:_optimizeMemoryUsage()
   -- Trigger memory optimization in agents
   for _, agent in ipairs(self.registeredAgents) do
      agent.cognitiveKernel:optimize()
   end
end

function TaskOrchestrator:_optimizeBandwidthUsage()
   -- Reduce inter-agent communication temporarily
   -- Implementation would involve throttling message passing
end

return TaskOrchestrator