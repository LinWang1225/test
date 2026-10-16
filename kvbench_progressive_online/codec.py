# SPDX-License-Identifier: Apache-2.0
"""Bit-exact 2+1+1 unpackable planar packing of the original native INT4 codes.

One tile contains all packed high2, then all packed bit1, then all packed bit0.
For K [D,G], for V [G,D]; both dimensions are multiples of 8. No extra KV
allocation: (2+1+1) bits per code, including the whole tile.
"""
from __future__ import annotations
import numpy as np


def _check_array(a):
    if a.ndim < 2 or a.shape[-1] % 8:
        raise ValueError('Expected tile (..., rows, columns) with columns divisible by 8')
    if np.any(a < 0) or np.any(a > 15):
        raise ValueError('Original K4V4 codes must be within [0,15]')


def _bits_numpy(a, nbits):
    a=np.asarray(a,np.uint8)
    div=8//nbits
    m=(1<<nbits)-1
    a=a.reshape(*a.shape[:-1],a.shape[-1]//div,div)
    out=np.zeros(a.shape[:-1],np.uint8)
    for i in range(div):out|=(a[...,i]&m)<<(i*nbits)
    return out


def pack_numpy(c):
    c=np.asarray(c)
    _check_array(c)
    c=c.astype(np.uint8)
    shape=c.shape
    parts=[_bits_numpy(c>>2,2),_bits_numpy((c>>1)&1,1),_bits_numpy(c&1,1)]
    return np.concatenate([p.reshape(*shape[:-2],-1) for p in parts],axis=-1).reshape(*shape[:-1],shape[-1]//2)


def unpack_numpy(packed,cols,depth=4):
    p=np.asarray(packed,np.uint8)
    if depth not in (2,3,4) or p.ndim<2 or cols%8 or p.shape[-1]*2!=cols:
        raise ValueError('Invalid packed tile/precision')
    rows=p.shape[-2]
    flat=p.reshape(*p.shape[:-2],-1)
    n=rows*cols
    n2,n1=n//4,n//8
    def unpack(raw,bits,width):
        raw=raw.reshape(*p.shape[:-2],rows,cols//(8//bits))
        shifts=np.arange(8//bits,dtype=np.uint8)*bits
        return ((raw[...,None]>>shifts)&((1<<bits)-1)).reshape(*p.shape[:-2],rows,cols)
    h=unpack(flat[...,:n2],2,cols).astype(np.int32)
    if depth==2:return 4*h+1.5
    m=unpack(flat[...,n2:n2+n1],1,cols).astype(np.int32)
    if depth==3:return 4*h+2*m+0.5
    l=unpack(flat[...,n2+n1:n2+2*n1],1,cols).astype(np.int32)
    return (4*h+2*m+l).astype(np.uint8)


def _torch_plane(q,bits):
    div=8//bits
    shape=q.shape
    r=q.reshape(*shape[:-1],shape[-1]//div,div)
    v=r[...,0]
    for i in range(1,div): v=v | (r[...,i] << (bits*i))
    return v.reshape(*shape[:-2],-1)


def pack_211_torch(code):
    """Drop-in for KVarN store _pack_4bit and 4-bit _pack_lowbit."""
    import torch
    if code.ndim<2 or code.shape[-1]%8:
        raise ValueError('INT4 tile width must be divisible by eight')
    q=code.to(torch.uint8)&15
    hi=_torch_plane(q>>2,2)
    b1=_torch_plane((q>>1)&1,1)
    b0=_torch_plane(q&1,1)
    return torch.cat((hi,b1,b0),dim=-1).reshape(*q.shape[:-1],q.shape[-1]//2)


def _torch_unplane(raw,rows,cols,bits):
    import torch
    div=8//bits
    t=raw.reshape(*raw.shape[:-1],rows,cols//div)
    return torch.stack(tuple((t>>(bits*i))&((1<<bits)-1) for i in range(div)),dim=-1).reshape(*raw.shape[:-1],rows,cols)


def unpack_211_torch(packed,original_last_dim):
    """Drop-in for KVarN torch decode _unpack_4bit; full INT4 recovered."""
    import torch
    p=packed.to(torch.uint8)
    cols=original_last_dim
    if p.ndim<2 or cols%8 or p.shape[-1]*2!=cols:
        raise ValueError('Invalid 2+1+1 tile shape')
    rows=p.shape[-2]
    flat=p.reshape(*p.shape[:-2],-1)
    n=rows*cols
    hi=_torch_unplane(flat[...,:n//4],rows,cols,2)
    b1=_torch_unplane(flat[...,n//4:3*n//8],rows,cols,1)
    b0=_torch_unplane(flat[...,3*n//8:n//2],rows,cols,1)
    return ((hi<<2)|(b1<<1)|b0).to(torch.uint8)
