import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
from torch.backends import cudnn


def normalize_max(data):
    max_val = np.max(np.abs(data))
    return data / max_val if max_val != 0 else data

class MyDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as file:
            tabular = file['para_all'][:]
            label = file['label'][:]

            FA_v_UD = file['FA_v_UD'][:]
            FA_d_UD = file['FA_d_UD'][:]
            FA_a_UD = file['FA_a_UD'][:]
            FA_v_NS = file['FA_v_NS'][:]
            FA_d_NS = file['FA_d_NS'][:]
            FA_a_NS = file['FA_a_NS'][:]
            FA_v_EW = file['FA_v_EW'][:]
            FA_d_EW = file['FA_d_EW'][:]
            FA_a_EW = file['FA_a_EW'][:]



            data_d_ud = normalize_max(file['Data_d_UD'][:])
            data_v_ud = normalize_max(file['Data_v_UD'][:])
            data_a_ud = normalize_max(file['Data_a_UD'][:])
            data_d_ns = normalize_max(file['Data_d_NS'][:])
            data_v_ns = normalize_max(file['Data_v_NS'][:])
            data_a_ns = normalize_max(file['Data_a_NS'][:])
            data_d_ew = normalize_max(file['Data_d_EW'][:])
            data_v_ew = normalize_max(file['Data_v_EW'][:])
            data_a_ew = normalize_max(file['Data_a_EW'][:])

        data = np.array([data_a_ud, data_v_ud, data_d_ud, data_a_ns, data_v_ns, data_d_ns, data_a_ew, data_v_ew, data_d_ew], dtype='float32')
        fa = np.array([FA_a_UD, FA_v_UD, FA_d_UD, FA_a_NS, FA_v_NS, FA_d_NS, FA_a_EW, FA_v_EW, FA_d_EW], dtype='float32')
        tabular = np.array([tabular], dtype='float32')

        y = np.array(label, dtype='float32')
        y[y == 2] = 0

        self.wave_data = torch.from_numpy(data).float().permute(2, 0, 1)
        self.fa_data = torch.from_numpy(fa).float().permute(2, 0, 1)
        self.tabular_data = torch.from_numpy(tabular).float().permute(2, 0, 1)
        self.y_data = torch.from_numpy(y).float()
        self.len = self.y_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.wave_data[index], self.fa_data[index], self.tabular_data[index], self.y_data[index]


def setup_seed(seed):
    import random
    import numpy as np
    import torch
    from torch.backends import cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def train_one_epoch(model, data_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    for waves, fas, tabular, labels in data_loader:
        waves, fas, tabular, labels = waves.to(device), fas.to(device), tabular.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(waves, fas, tabular)
        labels = labels.squeeze(axis=1).long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)
    return running_loss / len(data_loader), correct_preds / total_preds


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    with torch.no_grad():
        for waves, fas, tabular, labels in data_loader:
            waves, fas, tabular, labels = waves.to(device), fas.to(device), tabular.to(device), labels.to(device)
            outputs = model(waves, fas, tabular)
            labels = labels.squeeze(axis=1).long()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * waves.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
    return running_loss / total_preds, correct_preds / total_preds


def test_with_results(model, data_loader, device, criterion):
    model.eval()
    all_labels, all_preds, all_outputs = [], [], []
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for waves, fas, tabular, labels in data_loader:
            waves, fas, tabular, labels = waves.to(device), fas.to(device), tabular.to(device), labels.to(device)
            outputs = model(waves, fas, tabular)
            labels = labels.squeeze(axis=1).long()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    return all_labels, all_preds, total_loss / total, correct / total, all_outputs
