from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from torch_geometric.loader import DataLoader
import pandas as pd
import torch
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel
from sklearnex import patch_sklearn, unpatch_sklearn
import pickle
import torch
import pandas as pd
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import DataLoader, Data
from tqdm import tqdm
from util import logger
patch_sklearn()


class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads,
                             hidden_channels, heads=heads)
        self.fc1 = nn.Linear(hidden_channels * heads, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(
            data.num_nodes, dtype=torch.long)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def graph_info(raw_graph):
    node_info = []
    nodes = set()
    edges = []
    raw_graph = raw_graph[:-1].split("],")
    for item in raw_graph:
        try:
            if not item[-1] == ']':
                item = item + ']'
        except Exception as e:
            return None, None    # item = item.split(':')

        node_info.append(item)
        nodes.add(item.split(':')[0].strip()[1:-1].split('.')[-1])
        dest = item.split(':')[1].strip()[1:-1].split(',')
        for dst in dest:
            dst = dst.strip()[1:-1].split('.')[-1]
            nodes.add(dst)

    nodes = list(nodes)
    for edge_info in node_info:
        src = edge_info.split(':')[0].strip()[1:-1].split('.')[-1]
        src_index = nodes.index(src)
        dest = edge_info.split(':')[1].strip()[1:-1].split(',')
        for dst in dest:
            dst = dst.strip()[1:-1].split('.')[-1]
            dst_index = nodes.index(dst)
            edges.append((src_index, dst_index))
    return nodes, edges


def get_node_features(nodes, tokenizer, model):
    model.eval()
    node_features = []
    for n in nodes:
        inputs = tokenizer(n, return_tensors='pt',
                           padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        node_features.append(
            outputs.last_hidden_state.mean(dim=1))
    return node_features


def pad_features(feature_list, desired_dim):
    num_nodes = len(feature_list)
    padded_features = torch.zeros((num_nodes, desired_dim))
    for i, features in enumerate(feature_list):
        if features.shape[1] >= desired_dim:
            padded_features[i] = torch.tensor(features[:desired_dim])
        else:
            padded_features[i, :len(features)] = torch.tensor(features)
    return padded_features


def graph_info(raw_graph):
    node_info = []
    nodes = set()
    edges = []
    raw_graph = raw_graph[:-1].split("],")
    for item in raw_graph:
        try:
            if not item[-1] == ']':
                item = item + ']'
        except Exception as e:
            return None, None    # item = item.split(':')

        node_info.append(item)
        nodes.add(item.split(':')[0].strip()[1:-1].split('.')[-1])
        dest = item.split(':')[1].strip()[1:-1].split(',')
        for dst in dest:
            dst = dst.strip()[1:-1].split('.')[-1]
            nodes.add(dst)

    nodes = list(nodes)
    for edge_info in node_info:
        src = edge_info.split(':')[0].strip()[1:-1].split('.')[-1]
        src_index = nodes.index(src)
        dest = edge_info.split(':')[1].strip()[1:-1].split(',')
        for dst in dest:
            dst = dst.strip()[1:-1].split('.')[-1]
            dst_index = nodes.index(dst)
            edges.append((src_index, dst_index))
    return nodes, edges


def split_dataset_old(data_loader, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    full_dataset = data_loader.dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    torch.manual_seed(random_seed)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=data_loader.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=data_loader.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=data_loader.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def pad_graph(graphs, work_dir):
    df = pd.read_csv(os.path.join(work_dir, 'bert_all_adjacency_23_30_24.csv'))
    info = df.iloc[:, 0]
    raw_labels = df['category'].apply(
        lambda x: 1 if x == "scamware" else 0).tolist()
    labels = []
    for i, raw_graph in tqdm(enumerate(info)):
        nodes, edges = graph_info(raw_graph)
        if nodes == None:
            continue
        labels.append(raw_labels[i])
    for i, graph in enumerate(graphs):
        graph.x = pad_features(graph.x, 768)
        graph.y = labels[i]
    return graphs


def pre_pro_adj_whole_slow(pfn):
    if pfn.endswith('.csv'):
        df = pd.read_csv(pfn)
    else:
        df = pd.read_csv('bert_all_adjacency_23_30_24.csv')
    info = df.iloc[:, 0]
    raw_labels = df['category'].apply(
        lambda x: 1 if x == "scamware" else 0).tolist()
    labels = []
    raw_ml_labels = df['label'].tolist()
    mu_labels = []
    all_nodes = []
    all_edges = []
    for i, raw_graph in tqdm(enumerate(info)):
        nodes, edges = graph_info(raw_graph)
        if nodes == None:
            continue
        all_nodes.append(nodes)
        all_edges.append(edges)
        labels.append(raw_labels[i])
        mu_labels.append(raw_ml_labels[i])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    graphs = []
    for nodes, edges in tqdm(zip(all_nodes, all_edges), total=len(all_nodes)):
        node_feature = get_node_features(nodes, tokenizer, model)
        edge_indexs = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = Data(x=node_feature, edge_index=edge_indexs)
        graphs.append(data)
    # pfn = '0704_graphs.pkl'
    # with open(pfn, 'wb') as f:
    #     pickle.dump(graphs, f)
    print(graphs[0], len(graphs), len(labels))
    for i, graph in enumerate(graphs):
        graph.x = pad_features(graph.x, 768)
        graph.y = labels[i]
    return graphs


def pre_pro_adj(flag_hotload=True, pfn=None):
    if flag_hotload:
        graphs_load = []
        work_dir = os.basepath(pfn)
        pfn = ''
        with open(pfn, 'rb') as f:
            graphs_load = pad_graph(pickle.load(f), work_dir)
        return graphs_load
    else:
        if pfn.endswith(".csv"):
            return pre_pro_adj_whole_slow(pfn)
        else:
            logger.warning("Error in the input fea generation")


def get_data(flag_hotload=True, pfn="/home/aibot/workspace/SquiDroidAgent/MAGIC/0707result"):
    graphs_load = pre_pro_adj(flag_hotload=flag_hotload, pfn=pfn)
    return graphs_load


def split_dataset(graphs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    total_size = len(graphs)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    test_graphs = [graphs[i] for i in test_indices]

    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def get_train_loader(flag_hotload=True, pfn="/home/aibot/workspace/SquiDroidAgent/MAGIC/0707result/0704_graphs.pkl"):
    data_loader = get_data(flag_hotload=flag_hotload, pfn=pfn)
    train_loader, val_loader, test_loader = split_dataset(data_loader)
    return train_loader


def get_val_loader(flag_hotload=True, pfn="/home/aibot/workspace/SquiDroidAgent/MAGIC/0707result"):
    data_loader = get_data(flag_hotload=flag_hotload, pfn=pfn)
    train_loader, val_loader, test_loader = split_dataset(data_loader)
    return val_loader


def get_test_loader(flag_single=False, flag_hotload=True, pfn="/home/aibot/workspace/SquiDroidAgent/MAGIC/0707result"):
    data_loader = get_data(flag_hotload=flag_hotload, pfn=pfn)
    if flag_single:
        test_loader =  DataLoader(data_loader, batch_size=1, shuffle=True)
    else:
        train_loader, val_loader, test_loader = split_dataset(data_loader)
    return test_loader


def val(model, val_loader, device):
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            val_data = val_data.to(device)
            val_out = model(val_data)
            val_loss += F.nll_loss(val_out, val_data.y).item()
            _, val_predicted = val_out.max(dim=1)
            val_correct += val_predicted.eq(val_data.y).sum().item()
            val_total += val_data.y.size(0)
    val_avg_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')


def train():
    train_loader = get_train_loader(
        pfn="/home/aibot/workspace/SquiDroidAgent/MAGIC/0707result")
    val_loader = get_val_loader(
        pfn="/home/aibot/workspace/SquiDroidAgent/MAGIC/0707result")
    device = 'cuda:1'
    in_channels = 768
    hidden_channels = 768
    out_channels = 2
    heads = 4
    model = GATClassifier(in_channels, hidden_channels,
                          out_channels, heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    print_interval = 1  # 每隔10个epoch打印一次
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for data in train_loader:
            data = data.to(device)  # 将数据移动到 CUDA 设备 1
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = out.max(dim=1)
            correct += predicted.eq(data.y).sum().item()
            total += data.y.size(0)
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        if epoch % print_interval == 0:
            print(
                f'Epoch [{epoch}/{num_epochs}], Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
            val(model, val_loader, device)
