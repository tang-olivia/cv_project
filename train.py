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
from supcon_loss import SupConLoss
from torch.nn import TripletMarginLoss
from sklearn.metrics import silhouette_score

class IndexDataset(Dataset):
    def __init__(self, length):
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return idx 
    
class Experiment():
    def __init__(self):
        self.set_seed()

        self.model_dim = 512
        self.temperature = 0.07
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.num_epochs = 200
        self.batch_size = 64
        self.num_classes = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = '/om/user/oliviat/6.S058_project/model/fixed_weights.pt'
        
        self.model = JointEmbedTransformer(image_dim=1536, sequence_dim=1536, model_dim=self.model_dim)
        self.classifier = nn.Linear(self.model_dim, self.num_classes)
        self.criterion = SupConLoss(temperature=self.temperature)
        self.optimizer = torch.optim.AdamW(list(self.model.parameters()), 
                                          lr=self.learning_rate, weight_decay=self.weight_decay)
        self.tm_loss = TripletMarginLoss(margin=0.3, p=2)

    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def sample_embeddings(self, embeddings):
        indices = torch.randperm(embeddings.size(0))[:300]
        return embeddings[indices]

    def load_data(self):
        self.subcell_train, self.subcell_val = {}, {}
        self.uniprot_train, self.uniprot_val = {}, {}

        for i in range(self.num_classes):
            subcell_emb = self.sample_embeddings(torch.load(f'/om/user/oliviat/6.S058_project/model_data/subcell_embed_{i}.pt'))
            uniprot_emb = self.sample_embeddings(torch.load(f'/om/user/oliviat/6.S058_project/model_data/uniprot_embed_{i}.pt'))

            indices = torch.randperm(300)
            train_idx, val_idx = indices[:250], indices[250:]
            self.subcell_train[i] = subcell_emb[train_idx]
            self.subcell_val[i] = subcell_emb[val_idx]
            self.uniprot_train[i] = uniprot_emb[train_idx]
            self.uniprot_val[i] = uniprot_emb[val_idx]
        
        train_dataset = IndexDataset(length=self.num_classes * 4000)
        val_dataset = IndexDataset(length=self.num_classes * 750)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=self.collate_train)
        self.eval_loader = DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=self.collate_val)
    
    def collate_train(self, batch):
        batch_size = len(batch)
        image_embeddings, sequence_embeddings, labels = [], [], []
        sampled_classes = random.choices(range(self.num_classes), k=batch_size)

        for sampled_class in sampled_classes:
            image_embs = self.subcell_train[sampled_class]
            sequence_embs = self.uniprot_train[sampled_class]

            image_idx = random.randint(0, image_embs.shape[0] - 1)
            sequence_idx = random.randint(0, sequence_embs.shape[0] - 1)

            image_embeddings.append(image_embs[image_idx])
            sequence_embeddings.append(sequence_embs[sequence_idx])
            labels.append(sampled_class)
        
        image_embeddings = torch.stack(image_embeddings)
        sequence_embeddings = torch.stack(sequence_embeddings)
        batch_labels = torch.tensor(labels, dtype=torch.long)

        return image_embeddings, sequence_embeddings, batch_labels
    
    def collate_val(self, batch):
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

    def hard_neg_loss(self, joint_embeddings, labels):
        sim_matrix = torch.matmul(joint_embeddings, joint_embeddings.T)
        labels_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        neg_mask = ~labels_mask
        sim_matrix_neg = sim_matrix.masked_fill(~neg_mask, float('-inf'))
        hardest_neg_idx = sim_matrix_neg.argmax(dim=1)
        sim_matrix_pos = sim_matrix.masked_fill(labels_mask ^ torch.eye(joint_embeddings.size(0), dtype=bool, device=self.device),
                                                float('-inf'))
        hardest_pos_idx = sim_matrix_pos.argmax(dim=1)

        anchor = joint_embeddings
        positive = joint_embeddings[hardest_pos_idx]
        negative = joint_embeddings[hardest_neg_idx]
        return self.tm_loss(anchor, positive, negative)
    
    def train(self):
        self.model.train()
        self.classifier.train()
        
        max_val = float('-inf')
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.num_epochs):
                total_loss = 0.0

                train_embeddings = []
                train_labels = []

                for image_emb, sequence_emb, labels in self.train_loader:
                    image_emb, sequence_emb, labels = image_emb.to(self.device), sequence_emb.to(self.device), labels.to(self.device)

                    joint_emb = self.model(image_emb, sequence_emb)
                    joint_emb = nn.functional.normalize(joint_emb, dim=1)

                    train_embeddings.append(joint_emb.detach().cpu().numpy())
                    train_labels.append(labels.detach().cpu().numpy())

                    contrastive_loss = self.criterion(joint_emb.unsqueeze(1), labels)
                    logits = self.classifier(joint_emb)
                    ce_loss = F.cross_entropy(logits, labels)
                    hard_neg_loss = self.hard_neg_loss(joint_emb, labels)
                    loss = contrastive_loss + 0.3 * ce_loss + 0.7 * hard_neg_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                
                epoch_loss = total_loss / len(self.train_loader)
                val_score = self.eval()

                train_embeddings = np.concatenate(train_embeddings, axis=0)
                train_labels = np.concatenate(train_labels, axis=0)
                train_silhouette = silhouette_score(train_embeddings, train_labels)

                if val_score > max_val:
                    max_val = val_score
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    print(f'saved new best model at loss={epoch_loss:.4f} & silhouette score={val_score:.4f}')
                
                print(f'epoch {epoch+1} loss={epoch_loss:.4f}, train silhouette={train_silhouette:.4f}, val silhouette={val_score:.4f}')
    
    def eval(self):
        self.model.eval()
        self.classifier.eval()
        
        all_embeddings = []
        all_labels = []

        for image_emb, sequence_emb, labels in self.eval_loader:
            image_emb, sequence_emb, labels = image_emb.to(self.device), sequence_emb.to(self.device), labels.to(self.device)
            joint_emb = self.model(image_emb, sequence_emb)
            joint_emb = nn.functional.normalize(joint_emb, dim=1)

            all_embeddings.append(joint_emb.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0) 
        all_labels = np.concatenate(all_labels, axis=0)

        return silhouette_score(all_embeddings, all_labels)

experiment = Experiment()
experiment.model.to(experiment.device)
print('loading data...')
experiment.load_data()
print('starting training...')
experiment.train()
print('training completed!')
