from .mtcnn.detect_face_with_mtcnn import load_mtcnn_graph, detect_faceAPI
from .landmark68.detect_lm68 import load_lm_graph, detect_68p
from .face_parsing.face_parsing import load_face_parsing, get_seg_img
from .dml_csr.face_parsing import load_face_parsingV2, get_seg_imgV2
from .face_box import face_box
from .skin_mask.skin_mask import skinmask
from easydict import EasyDict as edict


class MTCNN_API:

    def __init__(self, model_path):
        '''
        Args:
            model_path: str. The pretrained mtcnn model.
        '''

        mtcnn_sess, pnet, rnet, onet = load_mtcnn_graph(model_path)
        self.mtcnn_sess = mtcnn_sess
        self.pnet = pnet
        self.rnet = rnet
        self.onet = onet

    def __call__(self, img):
        '''
        Args:
            img: numpy.array, float, (h, w, 3). The input image.
        Returns:
            five_points: numpy.array, float, (5, 2). The coordinates of 5 landmarks.
        '''
        five_points = detect_faceAPI(img, self.pnet, self.rnet, self.onet)
        return five_points

class FaceBox_API:
    def __init__(self, model_path, device='cuda'):
        '''
        Args:
            model_path: str. The pretrained facebox landmark model.
            device: str. The device to use (e.g., 'cuda:0', 'cuda:1').
        '''
        self.facebox_model = face_box(
            edict({
                'iscrop': True,
                'detector': 'retinaface',
                'device': device,
                'model_path': model_path,
            })
        ).detector_5p

    def __call__(self, img):
        return self.facebox_model(img)

class Landmark68_API:

    def __init__(self, lm_detector_path, mtcnn_path, device='cuda'):
        '''
        Args:
            lm_detector_path: str. The pretrained landmark detector model.
            mtcnn_path: str. The pretrained face detector model.
            device: str. The device to use (e.g., 'cuda:0', 'cuda:1').
        '''

        lm_sess, input_op, output_op = load_lm_graph(lm_detector_path)
        self.lm_sess = lm_sess
        self.input_op = input_op
        self.output_op = output_op

        self.mtcnn_model = FaceBox_API(mtcnn_path, device=device)

    def __call__(self, img):
        '''
        Args:
            img: numpy.array, float, (h, w, 3). The input image.
        Returns:
            lm_68: numpy.array, float, (68, 2). The coordinates of 68 landmarks.
        '''
        five_points = self.mtcnn_model(img[:, :, ::-1])
        if five_points is None:
            five_points = self.mtcnn_model(img)

        # Not detect faces
        if five_points is None:
            return None

        lm_68 = detect_68p(img, five_points, self.lm_sess, self.input_op, self.output_op)
        return lm_68


class FaceParsing_API:

    def __init__(self, parsing_pth, resnet18_path, device):
        '''
        Args:
            parsing_pth: str. The pretrained face parsing model.
            resnet18_path: str. The pretrained resnet18 model.
            device: str. The device.
        '''

        self.device = device
        self.parsing_net = load_face_parsing(parsing_pth, resnet18_path, device)

    def __call__(self, img, require_part=('face')):
        '''
        Args:
            img: numpy.array, float, (h, w, 3). The input image.
            require_part: Dict{str}. The required parts of face.
                options: [background,skin,l_brow,r_brow,l_eye,r_eye,eye_g,l_ear,r_ear,ear_r,
                          nose,mouth,u_lip,l_lip,neck,neck_l,cloth,hair,hat]
        Returns:
            require_part_masks: Dict{numpy.array}, float, (h, w, 3). The mask of each required part.
            seg_result: numpy.array, int64, (h, w). The semantic segmentation results, 0~18.
        '''
        require_part_masks, seg_result = get_seg_img(self.parsing_net, img, require_part, self.device)
        return require_part_masks, seg_result


class FaceParsingV2_API:
    def __init__(self, parsing_pth, device):
        '''
        Args:
            parsing_pth: str. The pretrained face parsing model.
            device: str. The device.
        '''

        self.device = device
        self.parsing_net = load_face_parsingV2(parsing_pth, device)

    def __call__(self, img, require_part=('face')):
        '''
        Args:
            img: numpy.array, float, (h, w, 3). The input image.
            require_part: Dict{str}. The required parts of face.
                options: [background,skin,l_brow,r_brow,l_eye,r_eye,eye_g,l_ear,r_ear,ear_r,
                          nose,mouth,u_lip,l_lip,neck,neck_l,cloth,hair,hat]
        Returns:
            require_part_masks: Dict{numpy.array}, float, (h, w, 3). The mask of each required part.
            seg_result: numpy.array, int64, (h, w). The semantic segmentation results, 0~18.
        '''
        require_part_masks, seg_result = get_seg_imgV2(self.parsing_net, img, require_part, self.device)
        return require_part_masks, seg_result

class SkinMask_API:

    def __init__(self):
        pass

    def __call__(self, img, return_uint8=False):
        '''
        Args:
            img: numpy.array, float, (h, w, 3). The input image (RGB).
            return_uint8: Bool. If true, return uint8 image (0~255).
        Returns:
            skin_att: numpy.array, float, (h, w, 3). The output skin attention mask (0~1).
        '''
        return skinmask(img, return_uint8)
