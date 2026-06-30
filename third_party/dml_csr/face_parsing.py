import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from inplace_abn import InPlaceABN
from .networks.dml_csr import DML_CSR

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

interp_op = torch.nn.Upsample(
            size=(512, 512), 
            mode='bilinear', 
            align_corners=True
        )


def load_face_parsingV2(save_pth, device):
    '''face_parsing setup'''
    n_classes = 19
    net = DML_CSR(num_classes=n_classes, abn=InPlaceABN, trained=False)
    net.load_state_dict(torch.load(save_pth, weights_only=False, map_location=device))
    net.to(device)
    net.eval()

    return net

def get_seg_imgV2(net, input_img, require_part=('face'), device='cuda'):
    ori_h, ori_w, _ = input_img.shape
    ''' face_parsing '''
    with torch.no_grad():
        face_parsing_in = cv2.resize(input_img, (473, 473))
        face_parsing_in = face_parsing_in.astype(np.uint8)
        face_parsing_in_tensor = to_tensor(face_parsing_in)
        face_parsing_in_tensor = torch.unsqueeze(face_parsing_in_tensor, 0)
        face_parsing_in_tensor = face_parsing_in_tensor.to(device)
        face_parsing_out = net(face_parsing_in_tensor)
        face_parsing_out = interp_op(face_parsing_out)
        parsing = face_parsing_out.detach().cpu().numpy()[0]  #  19 * 512 x 512

    seg_result = np.transpose(parsing, [1, 2, 0])  #  512 x 512 x 19
    seg_result = cv2.resize(seg_result, (ori_w, ori_h))  #  h x w x 19
    seg_result = np.argmax(seg_result, axis=2)  # h x w

    part_idx = {
        'background': 0,
        'skin': 1,
        'nose': 2,
        'eye_g': 3,
        'l_eye': 4,
        'r_eye': 5,
        'l_brow': 6,
        'r_brow': 7,
        'l_ear': 8,
        'r_ear': 9,
        'mouth': 10,
        'u_lip': 11,
        'l_lip': 12,
        'hair': 13,
        'hat': 14,
        'ear_r': 15,
        'neck_l': 16,
        'neck': 17,
        'cloth': 18
    }

    if isinstance(require_part, str):
        require_part = [require_part]
    require_part_masks = {}
    for part in require_part:
        if part == 'face':
            part_src = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
            part_src = (np.reshape(part_src, (1, 19, 1))).astype(np.uint8)
            part_mask = cv2.remap(part_src, seg_result.astype(np.float32), np.zeros_like(seg_result, dtype=np.float32),
                                  cv2.INTER_LINEAR)
            part_mask = np.tile(part_mask[..., np.newaxis], (1, 1, 3)).astype(np.float32)
            require_part_masks[part] = part_mask
        else:
            idx = part_idx[part]
            part_src = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            part_src[idx] = 1
            part_src = (np.reshape(part_src, (1, 19, 1))).astype(np.uint8)
            part_mask = cv2.remap(part_src, seg_result.astype(np.float32), np.zeros_like(seg_result, dtype=np.float32),
                                  cv2.INTER_LINEAR)
            part_mask = np.tile(part_mask[..., np.newaxis], (1, 1, 3)).astype(np.float32)
            require_part_masks[part] = part_mask

    return require_part_masks, seg_result
