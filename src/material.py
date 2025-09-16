# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import traceback

import numpy as np
import torch

from . import util
from . import texture


######################################################################################
# .mtl material format loading / storing
######################################################################################

def load_mtl(fn, clear_ks=True):
    import re
    mtl_path = os.path.dirname(fn)

    # Read file
    with open(fn) as f:
        lines = f.readlines()

    # Parse materials
    materials = []
    for line in lines:
        split_line = re.split(' +|\t+|\n+', line.strip())
        prefix = split_line[0].lower()
        data = split_line[1:]
        if 'newmtl' in prefix:
            material = {'name': data[0]}
            materials += [material]
        elif materials:
            if any(k in prefix for k in ('bsdf', 'map_kd', 'map_ks', 'bump', 'map_ns', 'ns', 'ka', 'refl', 'map_pm', 'map_pr')):
                material[prefix] = data[0]
            else:
                try:
                    material[prefix] = torch.tensor(tuple(float(d) for d in data), dtype=torch.float32, device='cuda')
                except Exception as e:
                    print(e, traceback.format_exc())

    # Convert everything to textures. Our code expects 'kd' and 'ks' to be texture maps. So replace constants with 1x1 maps
    for mat in materials:
        if not 'bsdf' in mat:
            mat['bsdf'] = 'pbr'

        if 'map_kd' in mat:
            mat['kd'] = texture.load_texture2D(os.path.join(mtl_path, mat['map_kd']))
        else:
            mat['kd'] = texture.Texture2D(mat['kd'])

        if 'map_ks' in mat:
            mat['ks'] = texture.load_texture2D(os.path.join(mtl_path, mat['map_ks']), channels=3)
        else:
            mat['ks'] = texture.Texture2D(mat['ks'])

        if 'bump' in mat:
            mat['normal'] = texture.load_texture2D(os.path.join(mtl_path, mat['bump']), lambda_fn=lambda x: x * 2 - 1, channels=3)

        # Convert Kd from sRGB to linear RGB
        mat['kd'] = texture.srgb_to_rgb(mat['kd'])

        if clear_ks:
            # Override ORM occlusion (red) channel by zeros. We hijack this channel
            for mip in mat['ks'].getMips():
                mip[..., 0] = 0.0

    return materials


def save_mtl(fn, material):
    folder = os.path.dirname(fn)
    with open(fn, "w") as f:
        f.write("newmtl defaultMat\n")

        # 1) BSDF (if any)
        if material and material.get("bsdf"):
            f.write(f"bsdf {material['bsdf']}\n")

        # 2) Diffuse (Kd) + ambient (Ka)
        if material and material.get("kd"):
            f.write("map_Kd texture_kd.png\n")
            texture.save_texture2D(
                os.path.join(folder, "texture_kd.png"),
                texture.rgb_to_srgb(material["kd"])
            )
        else:
            f.write("Kd 1 1 1\n")
        # even if you had a map_Kd, define an ambient fallback
        f.write("Ka 0 0 0\n")

        # 3) Specular (Ks) + specular exponent (Ns)
        if material and material.get("ks"):
            f.write("map_Ks texture_ks.png\n")
            texture.save_texture2D(
                os.path.join(folder, "texture_ks.png"),
                material["ks"]
            )
        else:
            f.write("Ks 0 0 0\n")
        # default specular exponent
        f.write("Ns 0\n")

        # 4) Optical density (Ni)
        # if you want a non-default, you could add material.get("ni") here
        f.write("Ni 1\n")

        # 5) Transmission filter (Tf)
        f.write("Tf 1 1 1\n")

        # 6) Normal map (bump)
        if material and material.get("normal"):
            f.write("bump texture_n.png\n")
            texture.save_texture2D(
                os.path.join(folder, "texture_n.png"),
                material["normal"],
                lambda_fn=lambda x: (x + 1) * 0.5
            )


######################################################################################
# Merge multiple materials into a single uber-material
######################################################################################

def _upscale_replicate(x, full_res):
    x = x.permute(0, 3, 1, 2)
    x = torch.nn.functional.pad(x, (0, full_res[1] - x.shape[3], 0, full_res[0] - x.shape[2]), 'replicate')
    return x.permute(0, 2, 3, 1).contiguous()


def merge_materials(materials, texcoords, tfaces, mfaces):
    assert len(materials) > 0
    # for mat in materials:
    #     assert mat['bsdf'] == materials[0]['bsdf'], "All materials must have the same BSDF (uber shader)"
    #     assert ('normal' in mat) is ('normal' in materials[0]), "All materials must have either normal map enabled or disabled"

    # ─── PATCH START ─────────────────────────────────────────────────────────────
    # If some materials have a 'normal' map and some don't, inject
    # a flat 1×1 normal into those that lack it, so the assertion below passes.
    has_norm = ['normal' in m for m in materials]
    if any(has_norm) and not all(has_norm):
        for m in materials:
            if 'normal' not in m:
                dev = m['kd'].data.device
                # Flat normal (0,0,1) as a 1×1 texture:
                nm = torch.tensor([[[[0.0, 0.0, 1.0]]]], device=dev)
                m['normal'] = texture.Texture2D(nm)
    # ─── PATCH END ────────────────────────────────────────────────────────────────

    for mat in materials:
        assert mat['bsdf'] == materials[0]['bsdf'], "All materials must have the same BSDF (uber shader)"
        assert ('normal' in mat) is ('normal' in materials[0]), "All materials must have either normal map enabled or disabled"

    uber_material = {
        'name': 'uber_material',
        'bsdf': materials[0]['bsdf'],
    }

    textures = ['kd', 'ks', 'normal']

    # Find maximum texture resolution across all materials and textures
    max_res = None
    for mat in materials:
        for tex in textures:
            tex_res = np.array(mat[tex].getRes()) if tex in mat else np.array([1, 1])
            max_res = np.maximum(max_res, tex_res) if max_res is not None else tex_res

    # Compute size of compund texture and round up to nearest PoT
    full_res = 2 ** np.ceil(np.log2(max_res * np.array([1, len(materials)]))).astype(np.int64)

    # Normalize texture resolution across all materials & combine into a single large texture
    for tex in textures:
        if tex in materials[0]:
            tex_data = torch.cat(tuple(util.scale_img_nhwc(mat[tex].data, tuple(max_res)) for mat in materials),
                                 dim=2)  # Lay out all textures horizontally, NHWC so dim2 is x
            tex_data = _upscale_replicate(tex_data, full_res)
            uber_material[tex] = texture.Texture2D(tex_data)

    # Compute scaling values for used / unused texture area
    s_coeff = [full_res[0] / max_res[0], full_res[1] / max_res[1]]

    # Recompute texture coordinates to cooincide with new composite texture
    new_tverts = {}
    new_tverts_data = []
    for fi in range(len(tfaces)):
        matIdx = mfaces[fi]
        for vi in range(3):
            ti = tfaces[fi][vi]
            if not (ti in new_tverts):
                new_tverts[ti] = {}
            if not (matIdx in new_tverts[ti]):  # create new vertex
                new_tverts_data.append([(matIdx + texcoords[ti][0]) / s_coeff[1], texcoords[ti][1] / s_coeff[
                    0]])  # Offset texture coodrinate (x direction) by material id & scale to local space. Note, texcoords are (u,v) but texture is stored (w,h) so the indexes swap here
                new_tverts[ti][matIdx] = len(new_tverts_data) - 1
            tfaces[fi][vi] = new_tverts[ti][matIdx]  # reindex vertex

    return uber_material, new_tverts_data, tfaces
