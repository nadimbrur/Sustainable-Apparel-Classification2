from torch.utils.data import Dataset
import numpy as np
class CustomDataset(Dataset):
    def __init__(self, data, transform = None):
        self.data = list(data.values)
        self.transform = transform
        label = []
        image = []
        for i in self.data:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)



