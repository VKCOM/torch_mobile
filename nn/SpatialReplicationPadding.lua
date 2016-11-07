local SpatialReplicationPadding, parent =
   torch.class('nn.SpatialReplicationPadding', 'nn.Module')

function SpatialReplicationPadding:__init(pad_l, pad_r, pad_t, pad_b)
   parent.__init(self)
   self.pad_l = pad_l
   self.pad_r = pad_r or self.pad_l
   self.pad_t = pad_t or self.pad_l
   self.pad_b = pad_b or self.pad_l
end

function SpatialReplicationPadding:updateOutput(input)
   if input:dim() == 3 or input:dim() == 4 then
      input.nn.SpatialReplicationPadding_updateOutput(self, input)
   else
      error('input must be 3 or 4-dimensional')
   end
   return self.output
end

function SpatialReplicationPadding:updateGradInput(input, gradOutput)
    return nil
end

function SpatialReplicationPadding:accGradParameters(input, gradOutput, scale)
    return nil
end
