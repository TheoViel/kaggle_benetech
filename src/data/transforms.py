import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def blur_transforms(p=0.5, blur_limit=5):
    """
    Applies MotionBlur or GaussianBlur random with a probability p.
    Args:
        p (float, optional): probability. Defaults to 0.5.
        blur_limit (int, optional): Blur intensity limit. Defaults to 5.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.MotionBlur(always_apply=True),
            albu.GaussianBlur(always_apply=True),
        ],
        p=p,
    )


def color_transforms(p=0.5):
    """
    Applies RandomGamma or RandomBrightnessContrast random with a probability p.
    Args:
        p (float, optional): probability. Defaults to 0.5.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.RandomGamma(gamma_limit=(50, 150), always_apply=True),
            albu.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.2, always_apply=True
            ),
            albu.ChannelShuffle(always_apply=True),
            albu.ToGray(always_apply=True),
#             albu.HueSaturationValue(always_apply=True),<
        ],
        p=p,
    )


def distortion_transforms(p=0.5):
    """
    Applies ElasticTransform with a probability p.
    Args:
        p (float, optional): probability. Defaults to 0.5.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.ElasticTransform(
                alpha=1,
                sigma=5,
                alpha_affine=10,
                border_mode=cv2.BORDER_CONSTANT,
                always_apply=True,
            )
        ],
        p=p,
    )


def get_transfos(augment=True, resize=(256, 256), mean=0, std=1, strength=1):
    """
    Returns transformations. todo

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        mean (np array, optional): Mean for normalization. Defaults to 0.
        std (np array, optional): Standard deviation for normalization. Defaults to 1.

    Returns:
        albumentation transforms: transforms.
    """
    resize_aug = [
        albu.Resize(resize[0], resize[1])
#         albu.RandomResizedCrop (resize[0], resize[1], scale=(0.2, 1.), always_apply=True)
    ] if resize else []

    normalizer = albu.Compose(
        resize_aug
        + [
            #             albu.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        p=1,
    )

    if augment:
        if strength == 0:
            augs = [
#                 albu.HorizontalFlip(p=0.5),
            ]
        elif strength == 1:
            augs = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.1,
                    shift_limit=0.0,
                    rotate_limit=20,
                    p=0.5,
                ),
                color_transforms(p=0.5),
            ]
        elif strength == 2:
            augs = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.2,
                    shift_limit=0.2,
                    rotate_limit=30,
                    p=0.75,
                ),
                color_transforms(p=0.5),
                blur_transforms(p=0.25),
            ]
        elif strength == 3:
            augs = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.2,
                    shift_limit=0.2,
                    rotate_limit=5,
                    p=1.,
                ),
                color_transforms(p=1.),
                blur_transforms(p=0.25),
#                 distortion_transforms(p=0.25),
            ]
    else:
        augs = []

    return albu.Compose(augs + [normalizer])


def get_transfos_centernet(augment=True, resize=(256, 256), mean=0, std=1, strength=1):
    """
    Returns transformations. todo

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        mean (np array, optional): Mean for normalization. Defaults to 0.
        std (np array, optional): Standard deviation for normalization. Defaults to 1.

    Returns:
        albumentation transforms: transforms.
    """
    resize_aug = [
        albu.Resize(resize[0], resize[1])
#         albu.RandomResizedCrop (resize[0], resize[1], scale=(0.2, 1.), always_apply=True)
    ] if resize else []

    normalizer = albu.Compose(
        resize_aug
        + [
            #             albu.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        p=1,
    )

    if augment:
        if strength == 0:
            augs = []
        elif strength == 1:
            augs = [
                albu.HorizontalFlip(p=0.5),
            ]
        elif strength == 2:
            augs = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.2,
                    shift_limit=0.2,
                    rotate_limit=10,
                    p=0.5,
                ),
                color_transforms(p=0.25),
                blur_transforms(p=0.1),
            ]
        elif strength == 3:
            augs = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.25,
                    shift_limit=0.25,
                    rotate_limit=25,
                    p=0.75,
                ),
                color_transforms(p=0.5),
                blur_transforms(p=0.25),
            ]
    else:
        augs = []

    return albu.Compose(augs + [normalizer])
