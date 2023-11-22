#pragma once

#ifndef FLT_TYP
#ifdef FLT64
#define FLT_TYP double
#else
#define FLT_TYP float
#endif
#endif