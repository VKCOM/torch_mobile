local SpatialConvolutionMM, parent = torch.class('nn.SpatialConvolutionMM', 'nn.Module')

function SpatialConvolutionMM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.padW = padW
   self.padH = padH

   self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   
   self:reset()
end

function SpatialConvolutionMM:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)  
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function SpatialConvolutionMM:updateOutput(input)
   local output = input.nn.SpatialConvolutionMM_updateOutput(self, input)
   self:freeTensors(self)
   return output
end

function SpatialConvolutionMM:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialConvolutionMM_updateGradInput(self, input, gradOutput)
   end
end

function SpatialConvolutionMM:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolutionMM_accGradParameters(self, input, gradOutput, scale)
end
