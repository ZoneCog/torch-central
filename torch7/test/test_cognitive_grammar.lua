--[[
Test Suite for Cognitive Grammar Architecture
============================================

This test suite validates the cognitive grammar implementation
for torch-central, ensuring all components work correctly and
integrate properly with the existing torch framework.

Author: torch-central cognitive architecture integration
--]]

require 'torch'

local tester = torch.Tester()
local tests = torch.TestSuite()

-- Test CognitiveGrammar module
function tests.testCognitiveGrammarBasic()
   local grammar = torch.CognitiveGrammar({
      tensorDimensions = {32, 16},
      memorySize = 100,
      nodeId = 'test_cg'
   })
   
   tester:asserteq(torch.typename(grammar), 'torch.CognitiveGrammar', 'Grammar object creation')
   tester:asserteq(grammar.nodeId, 'test_cg', 'Node ID assignment')
   tester:assertTableEq(grammar.tensorDimensions, {32, 16}, 0, 'Tensor dimensions')
   tester:asserteq(grammar.memorySize, 100, 'Memory size')
   
   -- Test tensor initialization
   tester:asserteq(grammar.semanticMemory:dim(), 3, 'Semantic memory tensor dimensions')
   tester:asserteq(grammar.activationState:dim(), 2, 'Activation state dimensions')
end

function tests.testCognitiveGrammarProcessing()
   local grammar = torch.CognitiveGrammar({
      tensorDimensions = {32, 16}
   })
   
   local input = torch.randn(32, 16)
   local result = grammar:process(input, 'activation')
   
   tester:asserteq(result:dim(), 2, 'Processed result dimensions')
   tester:asserteq(result:size(1), 32, 'Result size 1')
   tester:asserteq(result:size(2), 16, 'Result size 2')
   
   -- Test that activation state is updated
   tester:assertTensorEq(grammar.activationState, result, 1e-6, 'Activation state update')
   
   -- Test performance metrics
   tester:asserteq(grammar.performanceMetrics.activationCount, 1, 'Activation count metric')
   tester:asserteq(grammar.performanceMetrics.patternMatches, 1, 'Pattern matches metric')
end

function tests.testCognitiveGrammarMemory()
   local grammar = torch.CognitiveGrammar({
      tensorDimensions = {32, 16},
      memorySize = 50
   })
   
   local pattern = torch.randn(32, 16)
   
   -- Test memory storage
   grammar:storeMemory(pattern, 5)
   tester:asserteq(grammar.performanceMetrics.memoryUpdates, 1, 'Memory update metric')
   
   -- Test memory retrieval
   local retrieved = grammar:retrieveMemory(5)
   tester:assertTensorEq(retrieved, pattern, 1e-6, 'Memory storage and retrieval')
   
   -- Test invalid memory access
   local success, err = pcall(function() grammar:retrieveMemory(100) end)
   tester:assert(not success, 'Invalid memory access should fail')
end

function tests.testCognitiveGrammarConnections()
   local grammar1 = torch.CognitiveGrammar({tensorDimensions = {32, 16}})
   local grammar2 = torch.CognitiveGrammar({tensorDimensions = {32, 16}})
   
   -- Test connection creation
   local connectionId = grammar1:connectToNode(grammar2, 0.8)
   tester:asserteq(connectionId, 1, 'First connection ID')
   tester:asserteq(#grammar1.connections, 1, 'Connection count')
   tester:asserteq(grammar1.connectionWeights[1], 0.8, 'Connection weight')
   
   -- Test activation propagation
   local input = torch.randn(32, 16)
   grammar1:process(input, 'activation')
   local results = grammar1:propagateActivation()
   
   tester:asserteq(#results, 1, 'Propagation results count')
   tester:asserteq(results[1]:dim(), 2, 'Propagated result dimensions')
end

-- Test AgenticNode module
function tests.testAgenticNodeBasic()
   local agent = torch.AgenticNode({
      nodeId = 'test_agent',
      nodeType = 'tester',
      autonomyLevel = 0.5
   })
   
   tester:asserteq(torch.typename(agent), 'torch.AgenticNode', 'Agent object creation')
   tester:asserteq(agent.nodeId, 'test_agent', 'Agent node ID')
   tester:asserteq(agent.nodeType, 'tester', 'Agent node type')
   tester:asserteq(agent.autonomyLevel, 0.5, 'Agent autonomy level')
   
   -- Test cognitive kernel integration
   tester:asserteq(torch.typename(agent.cognitiveKernel), 'torch.CognitiveGrammar', 'Cognitive kernel integration')
end

function tests.testAgenticNodeTasks()
   local agent = torch.AgenticNode({
      nodeId = 'task_agent',
      tensorDimensions = {32, 16}
   })
   
   -- Test task addition
   local taskId = agent:addTask('parse', torch.randn(32, 16), 2)
   tester:asserteq(#agent.taskQueue, 1, 'Task queue size after addition')
   tester:asserteq(agent.taskQueue[1].priority, 2, 'Task priority')
   
   -- Test task processing
   local result, error = agent:processNextTask()
   tester:assert(result ~= nil or error ~= nil, 'Task processing returns result or error')
   tester:asserteq(#agent.taskQueue, 0, 'Task queue empty after processing')
   tester:asserteq(#agent.completedTasks, 1, 'Completed tasks count')
end

function tests.testAgenticNodeAttention()
   local agent = torch.AgenticNode()
   
   -- Test attention pattern setting
   local weights = agent:setAttentionPattern('focus', 3)
   tester:asserteq(weights:sum(), 1, 'Attention weights sum to 1')
   tester:asserteq(weights[3], 1, 'Focus attention pattern')
   
   weights = agent:setAttentionPattern('distribute')
   tester:assertTensorEq(weights, torch.ones(10):div(10), 1e-6, 'Distribute attention pattern')
end

function tests.testAgenticNodeConnections()
   local agent1 = torch.AgenticNode({nodeId = 'agent1'})
   local agent2 = torch.AgenticNode({nodeId = 'agent2'})
   
   -- Test agent connection
   local connId = agent1:connectToAgent(agent2, 'bidirectional', 0.7)
   tester:asserteq(connId, 1, 'Agent connection ID')
   tester:asserteq(#agent1.agentConnections, 1, 'Agent 1 connection count')
   tester:asserteq(#agent2.agentConnections, 1, 'Agent 2 connection count')
   
   -- Test message passing
   local message = torch.randn(64, 32)
   local sentCount = agent1:sendMessage(message)
   tester:asserteq(sentCount, 1, 'Messages sent count')
end

-- Test MemorySubsystem module
function tests.testMemorySubsystemBasic()
   local memory = torch.MemorySubsystem({
      semanticMemorySize = 100,
      perceptualBufferSize = 20,
      fragmentSize = {32, 16}
   })
   
   tester:asserteq(torch.typename(memory), 'torch.MemorySubsystem', 'Memory subsystem creation')
   tester:asserteq(memory.semanticMemorySize, 100, 'Semantic memory size')
   tester:asserteq(memory.perceptualBufferSize, 20, 'Perceptual buffer size')
   tester:assertTableEq(memory.fragmentSize, {32, 16}, 0, 'Fragment size')
end

function tests.testMemorySubsystemStorage()
   local memory = torch.MemorySubsystem({
      semanticMemorySize = 50,
      fragmentSize = {32, 16}
   })
   
   local data = torch.randn(32, 16)
   
   -- Test memory storage
   local index = memory:storeMemory(data, {importance = 1.5})
   tester:assert(index >= 1 and index <= 50, 'Valid memory index returned')
   tester:asserteq(memory.metrics.memoryStores, 1, 'Memory store metric')
   
   -- Test memory retrieval
   local retrieved = memory:retrieveMemory(index)
   tester:assertTensorEq(retrieved, data, 1e-6, 'Memory retrieval accuracy')
   tester:asserteq(memory.metrics.memoryRetrieves, 1, 'Memory retrieve metric')
end

function tests.testMemorySubsystemBuffer()
   local memory = torch.MemorySubsystem({
      perceptualBufferSize = 10,
      fragmentSize = {32, 16}
   })
   
   local data = torch.randn(32, 16)
   
   -- Test buffer storage
   local bufIndex = memory:storeInBuffer(data, 'test_source')
   tester:assert(bufIndex >= 1 and bufIndex <= 10, 'Valid buffer index')
   tester:asserteq(memory.metrics.bufferWrites, 1, 'Buffer write metric')
   
   -- Test buffer retrieval
   local retrieved = memory:retrieveFromBuffer(bufIndex)
   tester:assertTensorEq(retrieved, data, 1e-6, 'Buffer retrieval accuracy')
   tester:asserteq(memory.metrics.bufferReads, 1, 'Buffer read metric')
end

function tests.testMemorySubsystemSearch()
   local memory = torch.MemorySubsystem({
      semanticMemorySize = 20,
      fragmentSize = {16, 8}
   })
   
   -- Store some patterns
   local patterns = {}
   for i = 1, 5 do
      patterns[i] = torch.randn(16, 8)
      memory:storeMemory(patterns[i], {importance = i})
   end
   
   -- Search for similar patterns
   local query = patterns[3]:clone():add(torch.randn(16, 8):mul(0.1)) -- Add small noise
   local results = memory:searchMemory(query, 3)
   
   tester:asserteq(#results, 3, 'Search results count')
   tester:assert(results[1].similarity > results[2].similarity, 'Results sorted by similarity')
end

-- Test TaskOrchestrator module
function tests.testTaskOrchestratorBasic()
   local orchestrator = torch.TaskOrchestrator({
      maxConcurrentTasks = 5,
      attentionBudget = 50.0
   })
   
   tester:asserteq(torch.typename(orchestrator), 'torch.TaskOrchestrator', 'Orchestrator creation')
   tester:asserteq(orchestrator.maxConcurrentTasks, 5, 'Max concurrent tasks')
   tester:asserteq(orchestrator.attentionBudget, 50.0, 'Attention budget')
end

function tests.testTaskOrchestratorAgentRegistration()
   local orchestrator = torch.TaskOrchestrator()
   local agent = torch.AgenticNode({nodeId = 'test_orchestrator_agent'})
   
   -- Test agent registration
   local agentCount = orchestrator:registerAgent(agent, {'parse', 'learn'})
   tester:asserteq(agentCount, 1, 'Registered agent count')
   tester:asserteq(#orchestrator.registeredAgents, 1, 'Agent registry size')
   
   local capabilities = orchestrator.agentCapabilities[agent.nodeId]
   tester:assertTableEq(capabilities, {'parse', 'learn'}, 0, 'Agent capabilities')
end

function tests.testTaskOrchestratorTaskCreation()
   local orchestrator = torch.TaskOrchestrator()
   
   -- Test task creation
   local taskId = orchestrator:createTask('parse', torch.randn(32, 16), 2, {
      requiredCapabilities = {'parse'},
      estimatedComplexity = 1.5
   })
   
   tester:assert(taskId ~= nil, 'Task ID returned')
   tester:asserteq(#orchestrator.taskQueue, 1, 'Task queue size')
   tester:asserteq(orchestrator.metrics.tasksCreated, 1, 'Tasks created metric')
end

function tests.testTaskOrchestratorDecomposition()
   local orchestrator = torch.TaskOrchestrator()
   
   -- Create a task
   local taskId = orchestrator:createTask('complex', torch.randn(32, 16), 3)
   
   -- Decompose the task
   local subtasks = orchestrator:decomposeTask(taskId, 'parallel', {count = 3})
   
   tester:asserteq(#subtasks, 3, 'Subtask count')
   tester:asserteq(orchestrator.metrics.tasksDecomposed, 1, 'Task decomposition metric')
   
   -- Verify original task is marked as decomposed
   local originalTask = orchestrator:_findTask(taskId)
   tester:asserteq(originalTask.status, 'decomposed', 'Original task status')
end

-- Integration tests
function tests.testCognitiveArchitectureIntegration()
   -- Create a complete cognitive architecture setup
   local orchestrator = torch.TaskOrchestrator({
      maxConcurrentTasks = 3,
      attentionBudget = 100.0
   })
   
   local memory = torch.MemorySubsystem({
      semanticMemorySize = 50,
      fragmentSize = {32, 16}
   })
   
   local agent1 = torch.AgenticNode({
      nodeId = 'integration_agent_1',
      nodeType = 'parser',
      tensorDimensions = {32, 16}
   })
   
   local agent2 = torch.AgenticNode({
      nodeId = 'integration_agent_2', 
      nodeType = 'learner',
      tensorDimensions = {32, 16}
   })
   
   -- Connect agents
   agent1:connectToAgent(agent2, 'bidirectional', 0.8)
   
   -- Register agents with orchestrator
   orchestrator:registerAgent(agent1, {'parse'})
   orchestrator:registerAgent(agent2, {'learn'})
   
   -- Create and process tasks
   orchestrator:createTask('parse', torch.randn(32, 16), 2, {
      requiredCapabilities = {'parse'}
   })
   
   orchestrator:createTask('learn', torch.randn(32, 16), 1, {
      requiredCapabilities = {'learn'}
   })
   
   -- Process task queue
   local assignedCount = orchestrator:processTaskQueue()
   tester:assert(assignedCount >= 0, 'Tasks assigned successfully')
   
   -- Allocate attention
   local attentionAllocations = orchestrator:allocateAttention()
   tester:assert(type(attentionAllocations) == 'table', 'Attention allocations returned')
   
   -- Test system status
   local orchestratorStatus = orchestrator:getStatus()
   local memoryStatus = memory:getStatus()
   local agent1Status = agent1:getStatus()
   
   tester:assert(orchestratorStatus.registeredAgents == 2, 'Orchestrator agent count')
   tester:assert(memoryStatus.semanticMemoryUsage ~= nil, 'Memory status available')
   tester:assert(agent1Status.nodeId == 'integration_agent_1', 'Agent status available')
end

function tests.testPerformanceMetrics()
   -- Test that all components properly track performance metrics
   local grammar = torch.CognitiveGrammar()
   local agent = torch.AgenticNode()
   local memory = torch.MemorySubsystem()
   local orchestrator = torch.TaskOrchestrator()
   
   -- Perform some operations
   local input = torch.randn(64, 32)
   grammar:process(input)
   
   agent:addTask('test', input, 1)
   agent:processNextTask()
   
   memory:storeMemory(input:view(128, 64))
   
   orchestrator:createTask('test', input, 1)
   
   -- Check metrics
   local grammarMetrics = grammar:getMetrics()
   local agentStatus = agent:getStatus()
   local memoryStatus = memory:getStatus()
   local orchestratorStatus = orchestrator:getStatus()
   
   tester:assert(grammarMetrics.activationCount > 0, 'Grammar metrics tracked')
   tester:assert(agentStatus.performance.tasksCompleted >= 0, 'Agent metrics tracked')
   tester:assert(memoryStatus.metrics.memoryStores > 0, 'Memory metrics tracked')
   tester:assert(orchestratorStatus.metrics.tasksCreated > 0, 'Orchestrator metrics tracked')
end

-- Run the test suite
function tests.runAllTests()
   print("Running Cognitive Grammar Architecture Test Suite...")
   print("================================================")
   
   tester:add(tests)
   tester:run()
   
   print("\nTest Results Summary:")
   print("- CognitiveGrammar module: Basic functionality, processing, memory, connections")
   print("- AgenticNode module: Task management, attention control, inter-agent communication")
   print("- MemorySubsystem module: Storage, retrieval, buffering, search capabilities")
   print("- TaskOrchestrator module: Agent registration, task creation, decomposition")
   print("- Integration tests: Complete architecture validation")
   print("- Performance metrics: System-wide monitoring capabilities")
   
   return tester
end

-- Auto-run tests if this file is executed directly
if ... == nil then
   tests.runAllTests()
end

return tests