#ifndef LUA_TEST
#define LUA_TEST

//pkg/torch/init.c
int luaopen_libtorch(lua_State *L);
//nn/init.c
int luaopen_libnn(lua_State *L);
#ifdef USE_BIN_COMPAT
//bincompat/bincompat.c
int luaopen_libbincompat(lua_State *L);
#endif

#endif
