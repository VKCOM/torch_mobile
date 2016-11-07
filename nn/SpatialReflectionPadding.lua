local SpatialReflectionPadding, parent =
   torch.class('nn.SpatialReflectionPadding', 'nn.Module')

function SpatialReflectionPadding:__init(pad_l, pad_r, pad_t, pad_b)
   parent.__init(self)
   self.pad_l = pad_l
   self.pad_r = pad_r or self.pad_l
   self.pad_t = pad_t or self.pad_l
   self.pad_b = pad_b or self.pad_l
end

function SpatialReflectionPadding:updateOutput(input)
   if input:dim() == 3 or input:dim() == 4 then
      input.nn.SpatialReflectionPadding_updateOutput(self, input)
   else
      error('input must be 3 or 4-dimensional')
   end

   self:freeTensors(self)

   return self.output
end

function SpatialReflectionPadding:updateGradInput(input, gradOutput)
   return nil
end

function SpatialReflectionPadding:accGradParameters(input, gradOutput, scale)
    return nil
end
