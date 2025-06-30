--[[
Memory Subsystem Module for torch-central
=========================================

This module implements the distributed memory subsystem for the cognitive
architecture. It manages semantic memory stores, perceptual buffers, and
distributed knowledge fragments across the hypergraph network.

Key Features:
- Distributed semantic memory storage
- Perceptual buffering and preprocessing
- Knowledge fragment management
- Memory synchronization protocols
- Activation spreading mechanisms

Author: torch-central cognitive architecture integration
--]]

local MemorySubsystem = torch.class('torch.MemorySubsystem')

function MemorySubsystem:__init(config)
   config = config or {}
   
   -- Memory configuration
   self.semanticMemorySize = config.semanticMemorySize or 2048
   self.perceptualBufferSize = config.perceptualBufferSize or 256
   self.fragmentSize = config.fragmentSize or {128, 64}
   self.maxFragments = config.maxFragments or 1000
   
   -- Initialize memory stores
   self.semanticMemory = torch.Tensor(self.semanticMemorySize, self.fragmentSize[1], self.fragmentSize[2]):zero()
   self.perceptualBuffer = torch.Tensor(self.perceptualBufferSize, self.fragmentSize[1], self.fragmentSize[2]):zero()
   
   -- Memory metadata
   self.memoryMetadata = {}
   self.bufferMetadata = {}
   self.fragmentIndex = {}
   
   -- Initialize metadata tables
   for i = 1, self.semanticMemorySize do
      self.memoryMetadata[i] = {
         id = 'mem_' .. i,
         creationTime = 0,
         lastAccess = 0,
         accessCount = 0,
         importance = 0,
         fragmented = false,
         associations = {}
      }
   end
   
   for i = 1, self.perceptualBufferSize do
      self.bufferMetadata[i] = {
         id = 'buf_' .. i,
         timestamp = 0,
         processed = false,
         sourceId = nil
      }
   end
   
   -- Activation spreading parameters
   self.activationThreshold = config.activationThreshold or 0.1
   self.decayRate = config.decayRate or 0.95
   self.spreadingRate = config.spreadingRate or 0.8
   
   -- Current activation levels
   self.activationLevels = torch.Tensor(self.semanticMemorySize):zero()
   
   -- Memory synchronization
   self.syncPeers = {}
   self.pendingSyncs = {}
   
   -- Performance metrics
   self.metrics = {
      memoryStores = 0,
      memoryRetrieves = 0,
      bufferWrites = 0,
      bufferReads = 0,
      fragmentations = 0,
      synchronizations = 0,
      activationSpreads = 0
   }
end

-- Store data in semantic memory
function MemorySubsystem:storeMemory(data, metadata)
   if not torch.isTensor(data) then
      error('Memory data must be a tensor')
   end
   
   -- Find available memory slot or least important occupied slot
   local targetIndex = self:_findMemorySlot()
   
   -- Ensure data fits fragment dimensions
   local processedData = self:_processDataForStorage(data)
   
   -- Store in memory
   self.semanticMemory[targetIndex]:copy(processedData)
   
   -- Update metadata
   local currentTime = os.time()
   self.memoryMetadata[targetIndex] = {
      id = metadata and metadata.id or 'mem_' .. targetIndex .. '_' .. currentTime,
      creationTime = currentTime,
      lastAccess = currentTime,
      accessCount = 1,
      importance = metadata and metadata.importance or 1,
      fragmented = false,
      associations = metadata and metadata.associations or {},
      sourceData = metadata and metadata.sourceData or nil
   }
   
   -- Update metrics
   self.metrics.memoryStores = self.metrics.memoryStores + 1
   
   return targetIndex
end

-- Retrieve data from semantic memory
function MemorySubsystem:retrieveMemory(index, updateAccess)
   updateAccess = updateAccess == nil and true or updateAccess
   
   if index < 1 or index > self.semanticMemorySize then
      error('Memory index out of range: ' .. tostring(index))
   end
   
   local data = self.semanticMemory[index]:clone()
   
   if updateAccess then
      -- Update access metadata
      self.memoryMetadata[index].lastAccess = os.time()
      self.memoryMetadata[index].accessCount = self.memoryMetadata[index].accessCount + 1
      
      -- Increase importance based on access frequency
      self.memoryMetadata[index].importance = self.memoryMetadata[index].importance + 0.1
   end
   
   -- Update metrics
   self.metrics.memoryRetrieves = self.metrics.memoryRetrieves + 1
   
   return data
end

-- Store data in perceptual buffer
function MemorySubsystem:storeInBuffer(data, sourceId)
   if not torch.isTensor(data) then
      error('Buffer data must be a tensor')
   end
   
   -- Find next available buffer slot (circular buffer)
   local targetIndex = (self.metrics.bufferWrites % self.perceptualBufferSize) + 1
   
   -- Process and store data
   local processedData = self:_processDataForStorage(data)
   self.perceptualBuffer[targetIndex]:copy(processedData)
   
   -- Update metadata
   self.bufferMetadata[targetIndex] = {
      id = 'buf_' .. targetIndex .. '_' .. os.time(),
      timestamp = os.time(),
      processed = false,
      sourceId = sourceId
   }
   
   -- Update metrics
   self.metrics.bufferWrites = self.metrics.bufferWrites + 1
   
   return targetIndex
end

-- Retrieve data from perceptual buffer
function MemorySubsystem:retrieveFromBuffer(index)
   if index < 1 or index > self.perceptualBufferSize then
      error('Buffer index out of range: ' .. tostring(index))
   end
   
   local data = self.perceptualBuffer[index]:clone()
   
   -- Mark as processed
   self.bufferMetadata[index].processed = true
   
   -- Update metrics
   self.metrics.bufferReads = self.metrics.bufferReads + 1
   
   return data
end

-- Process buffer contents and move to semantic memory
function MemorySubsystem:processBuffer()
   local processedCount = 0
   
   for i = 1, self.perceptualBufferSize do
      local metadata = self.bufferMetadata[i]
      
      -- Process unprocessed buffer entries
      if metadata.timestamp > 0 and not metadata.processed then
         local bufferData = self:retrieveFromBuffer(i)
         
         -- Apply perceptual processing
         local processedData = self:_applyPerceptualProcessing(bufferData)
         
         -- Store in semantic memory if significant
         if processedData:norm() > self.activationThreshold then
            local memoryIndex = self:storeMemory(processedData, {
               sourceData = metadata,
               importance = math.min(processedData:norm() / 10, 2)
            })
            
            -- Trigger activation spreading
            self:spreadActivation(memoryIndex, processedData:norm())
         end
         
         processedCount = processedCount + 1
      end
   end
   
   return processedCount
end

-- Spread activation through memory network
function MemorySubsystem:spreadActivation(sourceIndex, initialActivation)
   initialActivation = initialActivation or 1.0
   
   -- Set initial activation
   self.activationLevels[sourceIndex] = initialActivation
   
   -- Spread to associated memories
   local associations = self.memoryMetadata[sourceIndex].associations
   
   for _, associatedIndex in ipairs(associations) do
      if associatedIndex >= 1 and associatedIndex <= self.semanticMemorySize then
         local spreadActivation = initialActivation * self.spreadingRate
         
         if spreadActivation > self.activationThreshold then
            self.activationLevels[associatedIndex] = 
               self.activationLevels[associatedIndex] + spreadActivation
         end
      end
   end
   
   -- Apply decay to all activations
   self.activationLevels:mul(self.decayRate)
   
   -- Update metrics
   self.metrics.activationSpreads = self.metrics.activationSpreads + 1
   
   return self.activationLevels:clone()
end

-- Create association between memory fragments
function MemorySubsystem:createAssociation(index1, index2, strength)
   strength = strength or 1.0
   
   if index1 < 1 or index1 > self.semanticMemorySize or 
      index2 < 1 or index2 > self.semanticMemorySize then
      error('Memory indices out of range')
   end
   
   -- Add bidirectional associations
   table.insert(self.memoryMetadata[index1].associations, index2)
   table.insert(self.memoryMetadata[index2].associations, index1)
   
   -- Update importance based on associations
   self.memoryMetadata[index1].importance = self.memoryMetadata[index1].importance + strength * 0.1
   self.memoryMetadata[index2].importance = self.memoryMetadata[index2].importance + strength * 0.1
   
   return true
end

-- Search memory based on pattern similarity
function MemorySubsystem:searchMemory(queryPattern, maxResults)
   maxResults = maxResults or 10
   
   if not torch.isTensor(queryPattern) then
      error('Query pattern must be a tensor')
   end
   
   local processedQuery = self:_processDataForStorage(queryPattern)
   local similarities = {}
   
   for i = 1, self.semanticMemorySize do
      if self.memoryMetadata[i].creationTime > 0 then
         local memoryData = self.semanticMemory[i]
         local similarity = self:_calculateSimilarity(processedQuery, memoryData)
         
         table.insert(similarities, {
            index = i,
            similarity = similarity,
            metadata = self.memoryMetadata[i]
         })
      end
   end
   
   -- Sort by similarity (descending)
   table.sort(similarities, function(a, b) return a.similarity > b.similarity end)
   
   -- Return top results
   local results = {}
   for i = 1, math.min(maxResults, #similarities) do
      table.insert(results, similarities[i])
   end
   
   return results
end

-- Add synchronization peer
function MemorySubsystem:addSyncPeer(peerMemorySystem, syncType)
   syncType = syncType or 'full'
   
   if torch.typename(peerMemorySystem) ~= 'torch.MemorySubsystem' then
      error('Sync peer must be a MemorySubsystem instance')
   end
   
   table.insert(self.syncPeers, {
      peer = peerMemorySystem,
      syncType = syncType,
      lastSync = 0,
      syncCount = 0
   })
   
   return #self.syncPeers
end

-- Synchronize with peer memory systems
function MemorySubsystem:synchronize()
   local syncCount = 0
   
   for _, peerInfo in ipairs(self.syncPeers) do
      local peer = peerInfo.peer
      
      if peerInfo.syncType == 'full' then
         -- Full synchronization (simplified)
         syncCount = syncCount + self:_performFullSync(peer)
      elseif peerInfo.syncType == 'incremental' then
         -- Incremental synchronization
         syncCount = syncCount + self:_performIncrementalSync(peer, peerInfo.lastSync)
      end
      
      peerInfo.lastSync = os.time()
      peerInfo.syncCount = peerInfo.syncCount + 1
   end
   
   self.metrics.synchronizations = self.metrics.synchronizations + syncCount
   
   return syncCount
end

-- Get memory subsystem status
function MemorySubsystem:getStatus()
   local activeMemories = 0
   local totalImportance = 0
   
   for i = 1, self.semanticMemorySize do
      if self.memoryMetadata[i].creationTime > 0 then
         activeMemories = activeMemories + 1
         totalImportance = totalImportance + self.memoryMetadata[i].importance
      end
   end
   
   return {
      semanticMemoryUsage = activeMemories .. '/' .. self.semanticMemorySize,
      averageImportance = activeMemories > 0 and totalImportance / activeMemories or 0,
      totalActivation = self.activationLevels:sum(),
      maxActivation = self.activationLevels:max(),
      syncPeerCount = #self.syncPeers,
      metrics = self.metrics
   }
end

-- Helper function to find available memory slot
function MemorySubsystem:_findMemorySlot()
   -- First, try to find empty slot
   for i = 1, self.semanticMemorySize do
      if self.memoryMetadata[i].creationTime == 0 then
         return i
      end
   end
   
   -- If no empty slots, find least important memory
   local minImportance = math.huge
   local targetIndex = 1
   
   for i = 1, self.semanticMemorySize do
      local importance = self.memoryMetadata[i].importance
      if importance < minImportance then
         minImportance = importance
         targetIndex = i
      end
   end
   
   return targetIndex
end

-- Helper function to process data for storage
function MemorySubsystem:_processDataForStorage(data)
   local processed
   
   if data:dim() == 2 and data:size(1) == self.fragmentSize[1] and data:size(2) == self.fragmentSize[2] then
      processed = data:clone()
   elseif data:nElement() == self.fragmentSize[1] * self.fragmentSize[2] then
      processed = data:view(self.fragmentSize[1], self.fragmentSize[2])
   else
      -- Resize/pad data to fit fragment size
      processed = torch.Tensor(self.fragmentSize[1], self.fragmentSize[2]):zero()
      local minRows = math.min(data:size(1) or data:nElement(), self.fragmentSize[1])
      local minCols = math.min(data:size(2) or 1, self.fragmentSize[2])
      
      if data:dim() == 1 then
         processed[{{1, minRows}, {1, 1}}]:copy(data[{{1, minRows}}]:view(-1, 1))
      else
         processed[{{1, minRows}, {1, minCols}}]:copy(data[{{1, minRows}, {1, minCols}}])
      end
   end
   
   return processed
end

-- Helper function to apply perceptual processing
function MemorySubsystem:_applyPerceptualProcessing(data)
   -- Simple perceptual processing: normalization and activation
   local processed = data:clone()
   
   -- Normalize
   if processed:norm() > 0 then
      processed:div(processed:norm())
   end
   
   -- Apply activation function
   processed:tanh()
   
   return processed
end

-- Helper function to calculate similarity between patterns
function MemorySubsystem:_calculateSimilarity(pattern1, pattern2)
   -- Cosine similarity
   local dot = torch.dot(pattern1:view(-1), pattern2:view(-1))
   local norm1 = pattern1:norm()
   local norm2 = pattern2:norm()
   
   if norm1 > 0 and norm2 > 0 then
      return dot / (norm1 * norm2)
   else
      return 0
   end
end

-- Helper functions for synchronization
function MemorySubsystem:_performFullSync(peer)
   -- Simplified full sync: exchange high-importance memories
   local syncCount = 0
   local importanceThreshold = 1.5
   
   for i = 1, self.semanticMemorySize do
      if self.memoryMetadata[i].importance > importanceThreshold then
         local data = self:retrieveMemory(i, false)
         peer:storeMemory(data, {
            importance = self.memoryMetadata[i].importance * 0.8, -- Reduce importance for copied memories
            sourceData = 'sync_' .. self.memoryMetadata[i].id
         })
         syncCount = syncCount + 1
      end
   end
   
   return syncCount
end

function MemorySubsystem:_performIncrementalSync(peer, lastSyncTime)
   -- Sync only memories created/modified since last sync
   local syncCount = 0
   
   for i = 1, self.semanticMemorySize do
      if self.memoryMetadata[i].creationTime > lastSyncTime or 
         self.memoryMetadata[i].lastAccess > lastSyncTime then
         local data = self:retrieveMemory(i, false)
         peer:storeMemory(data, {
            importance = self.memoryMetadata[i].importance * 0.9,
            sourceData = 'incr_sync_' .. self.memoryMetadata[i].id
         })
         syncCount = syncCount + 1
      end
   end
   
   return syncCount
end

return MemorySubsystem