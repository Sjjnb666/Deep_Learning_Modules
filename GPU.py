import torch
def get_device():
    """
    检测是否有可用的GPU资源
    :return: 返回可用的设备 'cuda' 或 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'