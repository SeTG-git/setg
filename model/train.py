import dgl
import numpy as np
import torch
from tqdm import tqdm
from util.loaddata import transform_graph


def batch_level_train(model, graphs, train_loader, optimizer, max_epoch, device, n_dim=0, e_dim=0):
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        torch.cuda.empty_cache()  # 释放显存~
        model.train()
        loss_list = []
        for i, batch in enumerate(train_loader):
        # for i, batch in tqdm(enumerate(train_loader)):
            # if i == 1499:
            #     print(1497)
            #     pass
            batch_g = [transform_graph(graphs[idx][0], n_dim, e_dim).to(device) for idx in batch]
            batch_g = dgl.batch(batch_g)
            # model.train()
            loss = model(batch_g)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            loss_list_no_nan = np.nan_to_num(loss_list)
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list_no_nan):.4f}")
        # torch.save(model.state_dict(), "/home/aibot/workspace/SquiDroidAgent/MAGIC/checkpoints/checkpoint-apks{}.pt".format(epoch))
    return model
