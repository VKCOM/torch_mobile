#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/BatchNormalization.c"
#else

int nn_(BatchNormalization_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *running_mean = luaT_getfieldcheckudata(L, 1, "running_mean", torch_Tensor);
  THTensor *running_var = luaT_getfieldcheckudata(L, 1, "running_var", torch_Tensor);
  THTensor *save_mean = luaT_getfieldcheckudata(L, 1, "save_mean", torch_Tensor);
  THTensor *save_std = luaT_getfieldcheckudata(L, 1, "save_std", torch_Tensor);
  int train = (int)(luaT_getfieldcheckboolean(L, 1, "train"));
  real momentum = luaT_getfieldchecknumber(L, 1, "momentum");
  real eps = luaT_getfieldchecknumber(L, 1, "eps");

  THTensor_(resizeAs)(output, input);
  long nInput = THTensor_(size)(input, 1);
  long f,n = THTensor_(nElement)(input) / nInput;

  #pragma omp parallel for
  for (f = 0; f < nInput; ++f) {
    THTensor *in = THTensor_(newSelect)(input, 1, f);
    THTensor *out = THTensor_(newSelect)(output, 1, f);

    real mean, invstd;

    if (train) {
      // compute mean per input
      accreal sum = 0;
      TH_TENSOR_APPLY(real, in, sum += *in_data;);

      mean = (real) sum / n;
      THTensor_(set1d)(save_mean, f, (real) mean);

      // compute variance per input
      sum = 0;
      TH_TENSOR_APPLY(real, in,
        sum += (*in_data - mean) * (*in_data - mean););

      if (sum == 0 && eps == 0.0) {
        invstd = 0;
      } else {
        invstd = (real) (1 / sqrt(sum/n + eps));
      }
      THTensor_(set1d)(save_std, f, (real) invstd);

      // update running averages
      THTensor_(set1d)(running_mean, f,
        (real) (momentum * mean + (1 - momentum) * THTensor_(get1d)(running_mean, f)));

      accreal unbiased_var = sum / (n - 1);
      THTensor_(set1d)(running_var, f,
        (real) (momentum * unbiased_var + (1 - momentum) * THTensor_(get1d)(running_var, f)));
    } else {
      mean = THTensor_(get1d)(running_mean, f);
      invstd = 1 / sqrt(THTensor_(get1d)(running_var, f) + eps);
    }

    // compute output
    real w = weight ? THTensor_(get1d)(weight, f) : 1;
    real b = bias ? THTensor_(get1d)(bias, f) : 0;

    TH_TENSOR_APPLY2(real, in, real, out,
      *out_data = (real) (((*in_data - mean) * invstd) * w + b););

    THTensor_(free)(out);
    THTensor_(free)(in);
  }

  return 1;
}

static const struct luaL_Reg nn_(BatchNormalization__) [] = {
  {"BatchNormalization_updateOutput", nn_(BatchNormalization_updateOutput)},
  {NULL, NULL}
};

static void nn_(BatchNormalization_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(BatchNormalization__), "nn");
  lua_pop(L,1);
}

#endif
