import hydra
import torch
import time
import datetime
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataset import MNISTDataset
from model import QCNN
from plotting import plot_history
from conf.structured_config import Config


@hydra.main(config_name="config", version_base=None)
def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO)
    train_dataloader = DataLoader(
        dataset = MNISTDataset(
            *cfg.data.values(),
            train=True
        ),
        batch_size=cfg.train.batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset = MNISTDataset(
            *cfg.data.values(),
            train=False
        ),
        batch_size=2*cfg.data.min_length,
    )

    losses, scores = [], []

    model = QCNN(cfg)

    opt = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.train.lr
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=cfg.train.start_factor,
        end_factor=cfg.train.end_factor,
        total_iters=cfg.train.epoch_count
    )

    start = time.time()

    for epoch in range(cfg.train.epoch_count):
        # fit
        epoch_history = []
        for i, (x, y) in enumerate(train_dataloader):
            opt.zero_grad()
            loss = model(x, y)
            loss.backward()
            opt.step()
            end = str(datetime.timedelta(seconds=time.time()-start))
            epoch_history.append(loss[0])
        
        scheduler.step()
        loss = sum(epoch_history) / len(epoch_history)

        # predict
        x, y = next(iter(test_dataloader))
        y_pred = model.predict(x)
        score = accuracy_score(y, y_pred)
        end = str(datetime.timedelta(seconds=time.time()-start))

        losses.append(float(loss))
        scores.append(score)

        logging.info(
            f"epoch: {epoch+1:2d} | loss: {loss:.3f} | "
            f"accuracy: {score:.3f} | current time: {end:.7}"
        )

    plot_history(cfg.plot.path, losses, scores)
    logging.info(f"History has been plotted to {cfg.plot.path}")


if __name__ == "__main__":
    main()
