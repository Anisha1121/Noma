# train_gnn.py
# Training loop for link prediction on your NOMA graphs
# ---------------------------------------------------------------

import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.metrics import roc_auc_score

from model_gnn import PairPredictionGNN

# ---------------------------------------
# Config (default paths match your setup)
# ---------------------------------------
DEFAULT_DATA_PATH = "D:/Developer/NOMA_new/dataset/bpf_graph_dataset.pt"
DEFAULT_SAVE_DIR  = "D:/Developer/NOMA_new/dataset/checkpoints"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_edge_index_for_message_passing(data):
    """
    For message passing, we can use the graph's positive edges as connectivity.
    Ensure it's undirected for SAGEConv.
    """
    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    return edge_index

def get_pos_neg_batches(data, num_neg=None):
    """
    Returns positive and negative edges (2,E_pos), (2,E_neg) for link prediction.
    """
    pos_edge_index = data.edge_index  # positives are the NOMA pairs in this graph
    if num_neg is None:
        num_neg = pos_edge_index.size(1)

    # Negative sampling avoids existing edges
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=num_neg,
        method='sparse'
    )
    return pos_edge_index, neg_edge_index

def compute_loss(pos_logits, neg_logits):
    """
    BCE with logits on concatenated pos/neg edges.
    """
    pos_labels = torch.ones_like(pos_logits)
    neg_labels = torch.zeros_like(neg_logits)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels), logits.sigmoid().detach().cpu().numpy(), labels.detach().cpu().numpy()

def evaluate(model, loader, device):
    model.eval()
    losses = []
    all_scores, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # message passing graph
            mp_edge_index = build_edge_index_for_message_passing(data)

            pos_edge_index, neg_edge_index = get_pos_neg_batches(data)
            pos_logits, neg_logits, _ = model(data.x, mp_edge_index, pos_edge_index, neg_edge_index)
            loss, scores, labels = compute_loss(pos_logits, neg_logits)
            losses.append(loss.item())
            all_scores.append(scores)
            all_labels.append(labels)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = float('nan')
    return np.mean(losses), auc

def train(args):
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------------
    # Load prebuilt graph dataset
    # -----------------------------
    # Fix for PyTorch 2.6+ security: allow PyTorch Geometric objects
    import torch_geometric.data.data
    torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
    graphs = torch.load(args.data_path, weights_only=False)
    print(f"Loaded {len(graphs)} graphs from {args.data_path}")

    # -----------------------------
    # Train/Val/Test split by graphs
    # -----------------------------
    n_total = len(graphs)
    n_train = int(0.7 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        graphs,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=1, shuffle=False)

    # --------------------------------
    # Model / Optimizer / Device
    # --------------------------------
    # Infer input feature dimension from first graph
    in_channels = graphs[0].x.size(1)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = PairPredictionGNN(
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        out_channels=args.out_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_auc = -1.0
    best_path = os.path.join(args.save_dir, "best_model.pt")

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for data in train_loader:
            data = data.to(device)

            mp_edge_index = build_edge_index_for_message_passing(data)
            pos_edge_index, neg_edge_index = get_pos_neg_batches(data, num_neg=None)

            pos_logits, neg_logits, _ = model(data.x, mp_edge_index, pos_edge_index, neg_edge_index)
            loss, _, _ = compute_loss(pos_logits, neg_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        val_loss, val_auc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | "
              f"TrainLoss {np.mean(train_losses):.4f} | "
              f"ValLoss {val_loss:.4f} | "
              f"ValAUC {val_auc:.4f}")

        # Save best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_state': model.state_dict(),
                'in_channels': in_channels,
                'hidden_dim': args.hidden_dim,
                'out_dim': args.out_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout
            }, best_path)
            print(f"  ✅ Saved best model to: {best_path}")

    # -----------------------------
    # Final test
    # -----------------------------
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    test_loss, test_auc = evaluate(model, test_loader, device)
    print("\n=== Final Test ===")
    print(f"TestLoss: {test_loss:.4f} | TestAUC: {test_auc:.4f}")
    print(f"Best checkpoint: {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    args = parser.parse_args()
    train(args)
