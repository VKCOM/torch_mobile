local ScalePixels, parent = torch.class('nn.ScalePixels', 'nn.Module')

--[[
-- mean_pixel is a Tensor of shape C, min and max are scalars
-- Maps an input tensor of shape (N, C, H, W) in the range [-1, 1]
-- to an output tensor of shape (N, C, H, W) where the cth channel
-- is in the range [min - mean_pixel[c], max - mean_pixel[c]]
--]

function ScalePixels:__init(mean_pixel, min, max)
  parent.__init(self)
  min = min or 0
  max = max or 255
  self.a = (max - min) / 2.0
  self.b = torch.mul(mean_pixel, -1):add((min + max / 2))
end


function ScalePixels:updateOutput(input)
  local b = self.b:view(1, -1, 1, 1):expandAs(input)
  self.output:mul(input, self.a):add(b)
  return self.output
end


function ScalePixels:updateGradInput(input, gradOutput)
  self.gradInput:mul(gradOutput, self.a)
  return self.gradInput
end

