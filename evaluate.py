import torch
import torch.nn as nn
import torch.nn.functional as F 
import random
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from transformer import JointEmbedTransformer
from sklearn.metrics import silhouette_score

class IndexDataset(Dataset):
    def __init__(self, length):
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return idx 

class Evaluation():
    def __init__(self):
        self.model_dim = 512
        self.num_classes = 5
        self.batch_size = 64

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = JointEmbedTransformer(image_dim=1536, sequence_dim=1536, model_dim=self.model_dim)
        self.model.load_state_dict(torch.load('/om/user/oliviat/6.S058_project/model/0.5409_weights.pt'))

    def set_seed(self, seed=0): # different seed for testing 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def sample_embeddings(self, embeddings):
        indices = torch.randperm(embeddings.size(0))[:300]
        return embeddings[indices]
    
    def sample_embeddings(self, embeddings):
        indices = torch.randperm(embeddings.size(0))[:300]
        return embeddings[indices]
    
    def load_data(self):
        self.subcell_val, self.uniprot_val = {}, {}

        for i in range(self.num_classes):
            self.subcell_val[i] = self.sample_embeddings(torch.load(f'/om/user/oliviat/6.S058_project/model_data/subcell_embed_{i}.pt'))
            self.uniprot_val[i] = self.sample_embeddings(torch.load(f'/om/user/oliviat/6.S058_project/model_data/uniprot_embed_{i}.pt'))

        val_dataset = IndexDataset(length=self.num_classes * 500)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
    def collate_fn(self, batch):
        batch_size = len(batch)
        image_embeddings, sequence_embeddings, labels = [], [], []
        sampled_classes = random.choices(range(self.num_classes), k=batch_size)

        for sampled_class in sampled_classes:
            image_embs = self.subcell_val[sampled_class]
            sequence_embs = self.uniprot_val[sampled_class]

            image_idx = random.randint(0, image_embs.shape[0] - 1)
            sequence_idx = random.randint(0, sequence_embs.shape[0] - 1)

            image_embeddings.append(image_embs[image_idx])
            sequence_embeddings.append(sequence_embs[sequence_idx])
            labels.append(sampled_class)
        
        image_embeddings = torch.stack(image_embeddings)
        sequence_embeddings = torch.stack(sequence_embeddings)
        batch_labels = torch.tensor(labels, dtype=torch.long)

        return image_embeddings, sequence_embeddings, batch_labels
    
    def plot_umap(self, embeddings, labels):
        label_map = {
            0: 'nucleus',
            1: 'endomembrane system',
            2: 'mitochondria', 
            3: 'plasma membrane',
            4: 'cytosol'
        }

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
        umap_embeddings = reducer.fit_transform(embeddings)
        labels = [label_map[label] for label in labels]

        plt.figure(figsize=(12, 10))
        scatter = sns.scatterplot(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            hue=labels,
            palette='tab10',
            s=20,
            legend='full'
        )
        plt.title('UMAP of Joint Embeddings')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')

        handles, labels_ = scatter.get_legend_handles_labels()
        plt.legend(handles=handles, labels=labels_, title='Class Labels', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, markerscale=2)

        plt.tight_layout()
        plt.savefig('/om/user/oliviat/6.S058_project/joint_emb_umap_clean.png', dpi=300)

    def eval(self):
        self.model.eval()

        embeddings = []
        all_labels = []

        for image_emb, sequence_emb, labels in self.val_loader:
            image_emb, sequence_emb, labels = image_emb.to(self.device), sequence_emb.to(self.device), labels.to(self.device)
            joint_emb = self.model(image_emb, sequence_emb)
            joint_emb = nn.functional.normalize(joint_emb, dim=1)

            embeddings.append(joint_emb.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        print(f'silhouette score={silhouette_score(embeddings, all_labels)}')
        self.plot_umap(embeddings, all_labels)

evaluation = Evaluation()
evaluation.model.to(evaluation.device)
evaluation.load_data()
evaluation.eval()