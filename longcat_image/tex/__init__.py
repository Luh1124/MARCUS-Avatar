import numpy as np
from scipy.io import loadmat

from utils import read_img
from .tex_func import remap_tex_from_input2D


class Tex_API:

    def __init__(self,
                 unwrap_info_path,
                 unwrap_info_mask_path,
                 unwrap_size=1024):
        '''
        Args:
            unwrap_info_path: str. The file path of unwrap information.
            unwrap_size: int. The image size of unwrap texture map.
        '''

        assert unwrap_size == 1024

        # template UV map
        # unwrap information
        unwrap_info = loadmat(unwrap_info_path)
        self.unwrap_uv_idx_bw = unwrap_info['uv_idx_bw'].astype(np.float32)
        self.unwrap_uv_idx_v_idx = unwrap_info['uv_idx_v_idx'].astype(np.float32)
        self.unwrap_info_mask = read_img(unwrap_info_mask_path, resize=(unwrap_size, unwrap_size), dst_range=1.)
        self.unwrap_uv_idx_bw = self.unwrap_uv_idx_bw * self.unwrap_info_mask
        self.unwrap_uv_idx_v_idx = self.unwrap_uv_idx_v_idx * self.unwrap_info_mask
        self.unwrap_size = unwrap_size

    def __call__(self, img, seg_mask, projXY, norm):
        '''
        Unwrap UV texture from input 2D image.

        Args:
            img: numpy.array (h, w, 3). The input 2D image.
            seg_mask: numpy.array (h, w, 3). The parsing mask of facial parts (without eyes and mouth).
            projXY: numpy.array (N, 2). The project XY coordinates (h x w) for each vertex of face.
            norm: numpy.array (N, 3). The normal vector for each vertex of face.
            Returns:
                unwrap_uv_tex: numpy.array (unwrap_size, unwrap_size, 3). The unwarpped UV texture map.
        '''

        # remap texture from input 2D image to UV map
        remap_tex, remap_mask = remap_tex_from_input2D(
                                                    input_img=img,
                                                    seg_mask=seg_mask,
                                                    projXY=projXY,
                                                    norm=norm,
                                                    unwrap_uv_idx_v_idx=self.unwrap_uv_idx_v_idx,
                                                    unwrap_uv_idx_bw=self.unwrap_uv_idx_bw
                                                )

        return remap_tex, remap_mask