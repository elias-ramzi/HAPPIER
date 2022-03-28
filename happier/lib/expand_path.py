import os


def expand_path(pth):
    pth = os.path.expandvars(pth)
    pth = os.path.expanduser(pth)

    if ('/local/DEEPLEARNING' in pth) and (not os.path.exists('/local/DEEPLEARNING')):
        if os.path.exists('/local/SSD_DEEPLEARNING_1'):
            pth = pth.replace('/local/DEEPLEARNING', '/local/SSD_DEEPLEARNING_1')

        elif os.getenv('SCRATCH'):
            pth = pth.replace('/local/DEEPLEARNING/image_retrieval', os.path.expandvars('$SCRATCH'))

    elif ('/share/DEEPLEARNING/datasets/image_retrieval' in pth) and (not os.path.exists('/share/DEEPLEARNING/datasets/image_retrieval')):
        if os.getenv('SCRATCH'):
            pth = pth.replace('/share/DEEPLEARNING/datasets/image_retrieval', os.path.expandvars('$SCRATCH'))

    return pth
