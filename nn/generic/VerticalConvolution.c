#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VerticalConvolution.c"
#else


static int nn_(VerticalConvolution_updateOutput)(lua_State *L)
{
   THTensor *input = luaT_checkudata(L, 2, torch_Tensor);

   int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
   int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
   int kL = luaT_getfieldcheckint(L, 1, "kL");

   THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
   THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
   THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

   luaL_argcheck(L, input->nDimension == 3 ||
                    input->nDimension == 4, 2, "3D or 4D (batch mode) tensor expected");

   // change to batch mode
   int batch = 1;
   if (input->nDimension == 3) {
      batch = 0;
      THTensor_(resize4d)(input, 1, nInputPlane, input->size[1], input->size[2]);
   }

   long batchSize    = input->size[0];
   long inputHeight  = input->size[2];
   long inputWidth   = input->size[3];
   long outputHeight = inputHeight - kL + 1;
   long outputWidth  = inputWidth;

   THTensor_(resize4d)(output, batchSize, nOutputPlane, outputHeight, outputWidth);

   int elt;
#pragma omp parallel for private(elt)
   for (elt = 0; elt < batchSize; elt++) {

      // select each batch
      THTensor *input_t  = THTensor_(newSelect)(input, 0, elt);
      THTensor *output_t = THTensor_(newSelect)(output, 0, elt);

      // fill biases
      int i, j, k;
      for (i = 0; i < nOutputPlane; i++) {
         THVector_(fill)(output_t->storage->data + output_t->storageOffset + output_t->stride[0]*i,
                         THTensor_(get1d)(bias, i), outputHeight*outputWidth);
      }

      // convolve vertically
      for (i = 0; i < nInputPlane; i++) {
         for (k = 0; k < kL; k++) {
            THVector_(add)(output_t->storage->data + output_t->storageOffset + output_t->stride[0]*i,
                           input_t->storage->data + input_t->storageOffset +
                           input_t->stride[0]*i + input_t->stride[1]*k,
                           *(THTensor_(data)(weight)+i*kL+k), outputHeight*outputWidth);
         }
      }

      // release temp tensors
      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
   }

   // revert to single batch
   if (batch == 0) {
      THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
      THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
   }

   return 1;
}


static int nn_(VerticalConvolution_updateGradInput)(lua_State *L)
{
   THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
   THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);

   int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
   int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
   int kL = luaT_getfieldcheckint(L, 1, "kL");

   THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
   THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

   THArgCheck(nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1,
              "Number of output features is not equal to nOutputPlane" );

   // change to batch mode
   int batch = 1;
   if (input->nDimension == 3) {
      batch = 0;
      THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
      THTensor_(resize4d)(gradOutput, 1, nOutputPlane, gradOutput->size[1], gradOutput->size[2]);
   }

   long batchSize    = input->size[0];
   long inputHeight  = input->size[2];
   long inputWidth   = input->size[3];
   long outputHeight = inputHeight - kL + 1;
   long outputWidth  = inputWidth;

   THTensor_(resizeAs)(gradInput, input);
   THTensor_(zero)(gradInput);

   int elt;
#pragma omp parallel for private(elt)
   for (elt = 0; elt < batchSize; elt++) {

      // select each batch
      THTensor *gradInput_t  = THTensor_(newSelect)(gradInput, 0, elt);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, elt);

      // convolve vertically
      int i, k;
      for (i = 0; i < nOutputPlane; i++) {
         for (k = 0; k < kL; k++) {
            THVector_(add)(gradInput_t->storage->data + gradInput_t->storageOffset +
                           gradInput_t->stride[0]*i + gradInput_t->stride[1]*k,
                           gradOutput_t->storage->data + gradOutput_t->storageOffset +
                           gradOutput_t->stride[0]*i,
                           *(THTensor_(data)(weight)+i*kL+k), outputHeight*outputWidth);
         }
      }

      // release temp tensors
      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
   }

   // revert to single batch
   if (batch == 0) {
      THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
      THTensor_(resize3d)(gradInput, nInputPlane, inputHeight, inputWidth);
      THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
   }

   return 1;
}


static int nn_(VerticalConvolution_accGradParameters)(lua_State *L)
{
   THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
   THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
   real scale = luaL_optnumber(L, 4, 1);
   int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
   int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
   int kL = luaT_getfieldcheckint(L, 1, "kL");

   THTensor *ones = luaT_getfieldcheckudata(L, 1, "ones", torch_Tensor);
   THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
   THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

   THArgCheck(nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1,
              "Number of output features is not equal to nOutputPlane" );

   // change to batch mode
   int batch = 1;
   if (input->nDimension == 3) {
      batch = 0;
      THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
      THTensor_(resize4d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
   }

   long batchSize    = input->size[0];
   long inputHeight  = input->size[2];
   long inputWidth   = input->size[3];
   long outputHeight = inputHeight - kL + 1;
   long outputWidth  = inputWidth;

   if (ones->nDimension != 1 || ones->size[0] < outputHeight*outputWidth) {
      THTensor_(resize1d)(ones, outputHeight*outputWidth);
      THTensor_(fill)(ones, 1);
   }

   int elt;
   for (elt = 0; elt < batchSize; elt++) {

      // select each batch in 2D
      THTensor *input_t      = THTensor_(newSelect)(input, 0, elt);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, elt);
      THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                   nOutputPlane, -1, outputWidth*outputHeight, -1);

      // dot products
      int i, k;
      for (i = 0; i < nInputPlane; i++) {
         for (k = 0; k < kL; k++) {
            *(gradWeight->storage->data + gradWeight->storageOffset + i*gradWeight->stride[0] + k) +=
               scale*THBlas_(dot)
                  (outputHeight*outputWidth,
                   gradOutput_t->storage->data + gradOutput_t->storageOffset + i*gradOutput_t->stride[0],
                   gradOutput_t->stride[2],
                   input_t->storage->data + input_t->storageOffset + i*input_t->stride[0] + k*outputWidth,
                   input_t->stride[2]);
         }
      }

      // fill biases
      THTensor_(addmv)(gradBias, 1, gradBias, scale, gradOutput2d, ones);

      THTensor_(free)(gradOutput2d);
      THTensor_(free)(input_t);
      THTensor_(free)(gradOutput_t);
   }

   // revert to single batch
   if (batch == 0) {
      THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
      THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
   }

   return 0;
}


static const struct luaL_Reg nn_(VerticalConvolution__) [] = {
  {"VerticalConvolution_updateOutput", nn_(VerticalConvolution_updateOutput)},
  {"VerticalConvolution_updateGradInput", nn_(VerticalConvolution_updateGradInput)},
  {"VerticalConvolution_accGradParameters", nn_(VerticalConvolution_accGradParameters)},
  {NULL, NULL}
};


static void nn_(VerticalConvolution_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(VerticalConvolution__), "nn");
  lua_pop(L,1);
}

#endif
