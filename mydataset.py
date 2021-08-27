from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.class_list = os.listdir(self.data_path)
        self.img_list=[]
        self.img_label=[]
        for list in self.class_list:
            list_name=list
            list=os.path.join(data_path,list)
            temp_image=os.listdir(list)
            self.img_list+=temp_image
            for item in temp_image:
                self.img_label.append(list_name)
    def __getitem__(self, index):
        img_title = self.img_list[index]
        img_label = self.img_label[index]
        img_path = os.path.join(self.data_path,img_label,img_title)
        img = Image.open(img_path)
        img=torchvision.transforms.ToTensor()(img)
        return img, img_label
    def __len__(self):
        return len(self.img_list)

# data=MyDataset(r"E:\Code\Experiment\Data\raw-ink-material-2021.5.31\Train-ResNet")
if __name__=="__main__":
    datas=[17.189245983958244, 17.79968310892582, 16.7098188996315, 20.099404722452164, 15.235986202955246, 18.202261209487915, 13.879845842719078, 16.998870629817247, 15.915361948311329, 11.775340855121613, 12.910257309675217, 12.307400472462177, 13.552596606314182, 12.846980698406696, 11.43618031591177, 11.631659843027592, 11.13261330127716, 12.314064495265484, 8.392442345619202, 13.924241483211517, 10.19830859452486, 9.944830525666475, 9.306077372282743, 10.251785777509212, 10.398503806442022, 8.640997368842363, 9.283821385353804, 9.056712243705988, 11.059935040771961, 9.935628034174442, 7.367854680866003, 7.345577508211136, 7.345770630985498, 8.646334851160645, 9.60602293536067, 6.634315762668848, 7.505976809188724, 7.93249829672277, 7.99958411604166]
    plt.plot(datas)
    plt.show()
