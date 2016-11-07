#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialReplicationPadding.c"
#else

static void nn_(SpatialReplicationPadding_updateOutput_frame)(
  real *input_p, real *output_p,
  long nslices,
  long iwidth, long iheight,
  long owidth, long oheight,
  int pad_l, int pad_r,
  int pad_t, int pad_b)
{
  int iStartX = fmax(0, -pad_l);
  int iStartY = fmax(0, -pad_t);
  int oStartX = fmax(0, pad_l);
  int oStartY = fmax(0, pad_t);

  long k, ip_x, ip_y;
#pragma omp parallel for private(k, ip_x, ip_y)
  for (k = 0; k < nslices; k++)
  {
    long i, j;
    for (i = 0; i < oheight; i++) {
      for (j = 0; j < owidth; j++) {
        if (j < pad_l) {
          ip_x = pad_l;
        } else if (j >= pad_l && j < iwidth + pad_l) {
          ip_x = j;
        } else {
          ip_x = iwidth + pad_l - 1;
        }
        ip_x = ip_x - oStartX + iStartX;

        if (i < pad_t) {
          ip_y = pad_t;
        } else if (i >= pad_t && i < iheight + pad_t) {
          ip_y = i;
        } else {
          ip_y = iheight + pad_t - 1;
        }
        ip_y = ip_y - oStartY + iStartY;

        real *dest_p = output_p + k*owidth*oheight + i * owidth + j;
        real *src_p = input_p + k*iwidth*iheight + ip_y * iwidth + ip_x;
        *dest_p = *src_p;
      }
    }
  }
}

int nn_(SpatialReplicationPadding_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int pad_l = luaT_getfieldcheckint(L, 1, "pad_l");
  int pad_r = luaT_getfieldcheckint(L, 1, "pad_r");
  int pad_t = luaT_getfieldcheckint(L, 1, "pad_t");
  int pad_b = luaT_getfieldcheckint(L, 1, "pad_b");

  int dimw = 2;
  int dimh = 1;
  int dimslices = 0;
  long nbatch = 1;
  long nslices;
  long iheight;
  long iwidth;
  long oheight;
  long owidth;
  real *input_data;
  real *output_data;

  THArgCheck(input->nDimension == 3 || input->nDimension == 4,
             2, "input must be 3 or 4-dimensional");

  if (input->nDimension == 4)
  {
    nbatch = input->size[0];
    dimw++;
    dimh++;
    dimslices++;
  }

  /* sizes */
  nslices = input->size[dimslices];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  oheight = iheight + pad_t + pad_b;
  owidth  = iwidth + pad_l + pad_r;

  THArgCheck(owidth >= 1 || oheight >= 1 , 2, "input is too small");

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  /* resize output */
  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);

    nn_(SpatialReplicationPadding_updateOutput_frame)(input_data, output_data,
                                                    nslices,
                                                    iwidth, iheight,
                                                    owidth, oheight,
                                                    pad_l, pad_r,
                                                    pad_t, pad_b);
  }
  else
  {
    long p;

    THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      nn_(SpatialReplicationPadding_updateOutput_frame)(
        input_data+p*nslices*iwidth*iheight,
        output_data+p*nslices*owidth*oheight,
        nslices,
        iwidth, iheight,
        owidth, oheight,
        pad_l, pad_r,
        pad_t, pad_b);
    }
  }

  /* cleanup */
  THTensor_(free)(input);

  return 1;
}

static const struct luaL_Reg nn_(SpatialReplicationPadding__) [] = {
  {"SpatialReplicationPadding_updateOutput", nn_(SpatialReplicationPadding_updateOutput)},
  {NULL, NULL}
};

static void nn_(SpatialReplicationPadding_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialReplicationPadding__), "nn");
  lua_pop(L,1);
}

#endif
