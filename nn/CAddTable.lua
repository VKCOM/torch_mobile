
local CAddTable, parent = torch.class('nn.CAddTable', 'nn.Module')

function CAddTable:__init(shi)
   parent.__init(self)
   self.gradInput = {}
   self.shi = shi or 0
end

function fitInput(input, s)
   local output = torch.Tensor()
   local N, C = input:size(1), input:size(2)
   local H, W = input:size(3), input:size(4)
   output:resize(N, C, H - 2 * s, W - 2 * s)
   output:copy(input[{{}, {}, {s + 1, H - s}, {s + 1, W - s}}])

   return output
end

function CAddTable:updateOutput(input)
   if self.shi > 0 then
      local base = fitInput(input[1], self.shi)
      self.output:resizeAs(base):copy(base)
   else
      self.output:resizeAs(input[1]):copy(input[1])
   end

   for i=2,#input do
      self.output:add(input[i])
      input[i] = nil
   end

   collectgarbage()
   return self.output
end

function CAddTable:updateGradInput(input, gradOutput)
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i])
      self.gradInput[i]:copy(gradOutput)
   end
   return self.gradInput
end
