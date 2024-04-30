import hydra
import torch
import time
import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataset import MNISTDataset
from model import QCNN
from plotting import plot_history


@hydra.main(config_path="conf", config_name="config", version_base="1.3.2")
def main(cfg):
    train_dataset = MNISTDataset(
        *cfg["data"].values(),
        train=True
    )
    
    test_dataset = MNISTDataset(
        *cfg["data"].values(),
        train=False
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["train"]["batch_size"]
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg["data"]["min_length"]
    )

    losses, scores = [], []

    model = QCNN()

    opt = torch.optim.Adam(
        model.parameters(), 
        lr=cfg["train"]["lr"]
    )

    start = time.time()

    for epoch in range(cfg["train"]["epoch_count"]):
        # fit
        print("_"*50)
        print("current parameters: \n")
        for name, param in model.state_dict().items():
            print(name, param, sep="\n")
        print("_"*50)
        epoch_history = []
        for i, (x, y) in enumerate(train_dataloader):
            opt.zero_grad()
            loss = model(x, y)
            loss.backward()
            opt.step()
            end = datetime.timedelta(seconds=time.time()-start)
            print(f"epoch: {epoch+1} | batch: {i+1:2d} | "
                  f"loss: {loss[0]:.3f} | current time: {end}")
            epoch_history.append(loss[0])
        
        loss = sum(epoch_history) / len(epoch_history)

        # predict
        x, y = next(iter(test_dataloader))
        y_pred = model.predict(x)
        score = accuracy_score(y, y_pred)

        losses.append(float(loss))
        scores.append(score)
        
        end = datetime.timedelta(seconds=time.time()-start)
        print(f"[*] epoch: {epoch+1:2d} | loss: {loss:.3f} | "
              f"accuracy: {score:.3f} | current time: {end}")

    plot_history(cfg["plot"]["path"], losses, scores)
    print("History has been plotted to", cfg["plot"]["path"])


if __name__ == "__main__":
    main()
