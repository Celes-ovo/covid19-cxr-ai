from AI_COVID19.init import *
import AI_COVID19.ImageProcess.SubModules.DataIO as daio


# RGB to Gray scale
def make_rgb_grey(nda, header):
    if len(nda.shape) == 3:
        if nda[..., 0].all() == nda[..., 1].all():
            new_nda = np.dot(nda[..., :3], [1, 0, 0])
        else:
            new_nda = np.dot(nda[..., :3], [0.2989, 0.5870, 0.1140])
        new_sitk = sitk.GetImageFromArray(new_nda)
        daio.header_writer(new_sitk, header)
        return new_nda, daio.header_extracter(new_sitk)
    elif len(nda.shape) == 2:
        return nda, header
    else:
        return False


# Zero Padding
def zero_padding(nda, header):
    size_sub = abs(nda.shape[0] - nda.shape[1])
    if size_sub > 0:
        if nda.shape[0] > nda.shape[1]:
            new_array = np.zeros(shape=(nda.shape[0], nda.shape[0]))
            for i in range(nda.shape[0]):
                for j in range(nda.shape[1]):
                    new_array[i, j + (size_sub//2)] = nda[i, j]
        elif nda.shape[0] < nda.shape[1]:
            new_array = np.zeros(shape=(nda.shape[1], nda.shape[1]))
            for i in range(nda.shape[0]):
                for j in range(nda.shape[1]):
                    new_array[i + (size_sub // 2), j] = nda[i, j]
        new_sitk = sitk.GetImageFromArray(new_array)
        daio.header_writer(new_sitk, header)
        return new_array, daio.header_extracter(new_sitk)
    else:
        return nda, header


# Resample
def resample_array(image_array, header_dic, spacing, mode, interpolator=sitk.sitkLinear):
    sitk_image = sitk.GetImageFromArray(image_array)
    daio.header_writer(sitk_image, header_dic)
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    if mode == '2d':
        new_spacing = [spacing, spacing, header_dic['spacing'][2]]
    elif mode == '3d':
        new_spacing = [spacing] * sitk_image.GetDimension()
    elif mode == 'custom':
        new_spacing = spacing
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in
                zip(original_size, original_spacing, new_spacing)]
    sitk_image = sitk.Resample(sitk_image, new_size, sitk.Transform(), interpolator, sitk_image.GetOrigin(), new_spacing,
                         sitk_image.GetDirection(), 0, sitk_image.GetPixelID())
    return sitk.GetArrayFromImage(sitk_image), daio.header_extracter(sitk_image)


# Resize
def resize_array(image_array, header_dic, size, mode, interpolator=sitk.sitkLinear):
    sitk_image = sitk.GetImageFromArray(image_array)
    daio.header_writer(sitk_image, header_dic)
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    new_size = list(original_size)
    if mode == '2d':
        new_size[0] = size[0]
        new_size[1] = size[1]
    elif mode == '3d':
        new_size[0] = size[0]
        new_size[1] = size[1]
        new_size[2] = size[2]
    else:
        print('!ERROR! Please input the correct mode \'2d\' or \'3d\'')
    new_spacing = [(ospc * osz / nsz) for osz, ospc, nsz in
                   zip(original_size, original_spacing, new_size)]
    sitk_image = sitk.Resample(sitk_image, new_size, sitk.Transform(), interpolator, sitk_image.GetOrigin(), new_spacing,
                         sitk_image.GetDirection(), 0, sitk_image.GetPixelID())
    return sitk.GetArrayFromImage(sitk_image), daio.header_extracter(sitk_image)


# Normalization
def norm_array(image_array):
    nda_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    return np.array(255*nda_image, dtype=np.uint8)


# Standardization
def stand_array(image_array):
    nda_image = (image_array - np.mean(image_array)) / np.std(image_array)
    return np.array(nda_image, dtype=np.float16)


# Adaptive Equalization
def adapequal_array(image_array, clip_val):
    """
    :param image_array: Numpy Array Image
    :param clip_val: Clipping Value (Default=0.1)
    :return:
    """
    rescale_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    if len(image_array.shape) == 2:
        rescale_image = exposure.equalize_adapthist(rescale_image, clip_limit=clip_val)
    elif len(image_array.shape) == 3:
        for i in range(rescale_image.shape[0]):
            aeh_img = exposure.equalize_adapthist(rescale_image[i], clip_limit=clip_val)
            rescale_image[i] = aeh_img
    else:
        print('!ERROR! please insert correct dimension (2d or 3d)')
    return np.array(rescale_image)