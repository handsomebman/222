import tifffile as tiff
import torch
import numpy as np

def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

def organized_pc_to_unorganized_pc_tensor(organized_pc):
    B, C, H, W = organized_pc.shape
    return organized_pc.reshape(B, C, -1)

def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img


def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).contiguous()
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='nearest')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0).contiguous()
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).contiguous().numpy()


def organized_pc_to_depth_map(organized_pc):
    return organized_pc[:, :, 2]


def replace_depth_in_organized_pc(organized_pc, depth_map):
    """
    Replace the depth channel (third channel, index 2) in organized_pc with the provided depth_map.
    
    Args:
        organized_pc: Organized point cloud array of shape (H, W, 3).
        depth_map: Depth map array of shape (H, W).
    
    Returns:
        New organized_pc with depth channel replaced.
    """
    if organized_pc.shape[:2] != depth_map.shape:
        raise ValueError("Shape mismatch: organized_pc height/width must match depth_map shape.")
    new_organized_pc = organized_pc.copy()
    new_organized_pc[:, :, 2] = depth_map
    return new_organized_pc


if __name__ == '__main__':
    # Example usage
    organized_pc = read_tiff_organized_pc('/root/shared-nvme/Datasets/mvtec_3d_anomaly_detection/bagel/train/good/xyz/000.tiff')
    resized_pc = resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=False)
    depth_map = organized_pc_to_depth_map(resized_pc)
    new_depth_map = np.ones_like(depth_map).astype(np.float32)
    new_organized_pc = replace_depth_in_organized_pc(resized_pc, new_depth_map)
    print("Original Organized PC shape:", organized_pc.shape)