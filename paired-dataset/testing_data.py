from PIL import Image
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch

trans = transforms.Compose([transforms.ToTensor()])
for i in range(901,1001):
    ground_truth = Image.open('D:/PYTHON/python_handle/paired-dataset/rainy_image_dataset/'
                              'testing/ground_truth/'+str(i)+'.jpg')
    ground_truth = trans(ground_truth)
    output = torch.FloatTensor(2, ground_truth.size(0), ground_truth.size(1), ground_truth.size(2),).fill_(0)
    output[0, :, :, :].copy_(ground_truth.data)
    output[1, :, :, :].copy_(ground_truth.data)
    vutils.save_image(output, 'rainy_dataset/testing/no_rain/' + str(i) + '.jpg', nrow=2,padding=0, normalize=True)
    for j in range(1,15):
        rainy_image = Image.open('D:/PYTHON/python_handle/paired-dataset/rainy_image_dataset/'
                                  'testing/rainy_image/' + str(i) + '_'+str(j)+'.jpg')
        rainy_image = trans(rainy_image)
        output[1, :, :, :].copy_(rainy_image.data)
        vutils.save_image(output, 'rainy_dataset/testing/'+ str(i) + '_'+str(j)+'.jpg', nrow=2, padding=0,normalize=True)

        mask = rainy_image - ground_truth
        mask_output = torch.FloatTensor(ground_truth.size(0), ground_truth.size(1), ground_truth.size(2),).fill_(0)
        mask_output.copy_(mask.data)
        vutils.save_image(mask, 'rainy_dataset/testing/mask/' + str(i) + '_' + str(j) + '.jpg', padding=0, normalize=True)
