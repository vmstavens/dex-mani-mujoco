

import numpy as np



def glFrustum_CD_float32(znear, zfar):
	zfar  = np.float32(zfar)
	znear = np.float32(znear)
	C = -(zfar + znear)/(zfar - znear)
	D = -(np.float32(2)*zfar*znear)/(zfar - znear)
	return C, D

def ogl_zbuf_projection(zlinear, C, D):
	zbuf = -C + (1/zlinear)*D # TODO why -C?
	return zbuf

def ogl_zbuf_projection_inverse(zbuf, C, D):
	zlinear = 1 / ((zbuf - (C)) / D) # TODO why -C?
	# zlinear = 1 / ((zbuf - (-C)) / D) # TODO why -C?
	return zlinear

def ogl_zbuf_default(zlinear, znear=None, zfar=None, C=None, D=None):
	if C is None:
		C, D = glFrustum_CD_float32(znear, zfar)
	zbuf = ogl_zbuf_projection(zlinear, C, D)
	zbuf_scaled = 0.5 * zbuf + 0.5
	return zbuf_scaled

def ogl_zbuf_negz(zlinear, znear=None, zfar=None, C=None, D=None):
	if C is None:
		C, D = glFrustum_CD_float32(znear, zfar)
		C = np.float32(-0.5)*C - np.float32(0.5)
		D = np.float32(-0.5)*D
	zlinear = ogl_zbuf_projection(zlinear, C, D)
	return zlinear

def ogl_zbuf_default_inv(zbuf_scaled, znear=None, zfar=None, C=None, D=None):
	if C is None:
		C, D = glFrustum_CD_float32(znear, zfar)
	zbuf = 2.0 * zbuf_scaled - 1.0
	zlinear = ogl_zbuf_projection_inverse(zbuf, C, D)
	return zlinear

def ogl_zbuf_negz_inv(zbuf, znear=None, zfar=None, C=None, D=None):
	if C is None:
		C, D = glFrustum_CD_float32(znear, zfar)
		C = np.float32(-0.5)*C - np.float32(0.5)
		D = np.float32(-0.5)*D
	zlinear = ogl_zbuf_projection_inverse(zbuf, C, D)
	return zlinear