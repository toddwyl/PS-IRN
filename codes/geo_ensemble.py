import numpy as np
import torch


def ensemble_augment(img):
    # ensemble img:
    # origin, flip, rot90, flip&rot90, rot180, flip&rot180, rot270, flip&rot270
    img_list = []
    img_list.append(img)

    if isinstance(img, np.ndarray):
        img_flip = img[::-1, :, :]
        img_list.append(img_flip)
        for i in range(1, 4):
            img_list.append(np.rot90(img, k=i))
            img_list.append(np.rot90(img_flip, k=i))

    elif isinstance(img, torch.Tensor):
        dims = [2, 3]
        img_flip = torch.flip(img, dims=[2])
        img_list.append(img_flip)
        for i in range(1, 4):
            img_list.append(torch.rot90(img, k=i, dims=dims))
            img_list.append(torch.rot90(img_flip, k=i, dims=dims))

    else:
        raise NotImplementedError("only support numpy and torch")

    return img_list


def ensemble2origin(img_list):
    # rot or flip back ensemble images to the origin position:
    # origin, flip, rot90, flip&rot90, rot180, flip&rot180, rot270, flip&rot270
    img2origin = []
    if isinstance(img_list[0], np.ndarray):
        for idx, img in enumerate(img_list):
            k = idx // 2
            if k > 0:
                img = np.rot90(img, k=-k)
            if idx % 2 == 1:
                img = img[::-1, :, :]
            img2origin.append(img)

    elif isinstance(img_list[0], torch.Tensor):
        for idx, img in enumerate(img_list):
            dims = [2, 3]
            k = idx // 2
            if k > 0:
                img = torch.rot90(img, k=-k, dims=dims)
            if idx % 2 == 1:
                img = torch.flip(img, dims=[2])
            img2origin.append(img)

    else:
        raise NotImplementedError("only support numpy and torch")

    return img2origin


# if __name__ == "__main__":
#     # test for ensemble_augment
#     img_test = "/home/ices/yl/SRDict/PS-IRN/codes/test_ense.png"
#     import cv2
#     img = cv2.imread(img_test)

#     imgs = ensemble_augment(np.array(img))
#     for i, im in enumerate(imgs):
#         cv2.imwrite(f'./ensemble_img/test_ense_cv2_np_{i}.png', im)
#     img_origins = ensemble2origin(imgs)
#     avg_img = np.sum(img_origins, axis=0) // len(img_origins)
#     print(avg_img)
#     cv2.imwrite('./ensemble_img/np_avg.png', avg_img)
#     for i, im in enumerate(img_origins):
#         cv2.imwrite(f'./ensemble_img/test_ense_cv2_origin_np_{i}.png', im)
    
#     imgs = ensemble_augment(torch.FloatTensor(img[None, ...]))
#     for i, im in enumerate(imgs):
#         cv2.imwrite(f'./ensemble_img/test_ense_cv2_{i}.png', im.numpy()[0].astype(np.uint8))
#     img_origins = ensemble2origin(imgs)
#     for i, im in enumerate(img_origins):
#         cv2.imwrite(f'./ensemble_img/test_ense_cv2_origin_{i}.png', im.numpy()[0].astype(np.uint8))