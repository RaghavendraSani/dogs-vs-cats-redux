from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

class Trainer:
    def __init__(self,model,train_loader,valid_loader,epochs=30,learning_rate=1e-4):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )

        self.scaler = torch.amp.GradScaler("cuda")

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(self.train_loader, start=1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).unsqueeze(1)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            running_loss += loss.item()
            if batch_idx % 50 == 0:
                print(
                    f"Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.5f}"
                )

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in self.valid_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).unsqueeze(1)

                with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                running_loss += loss.item()

        return running_loss / len(self.valid_loader)

    def train(self):
        best_loss = float("inf")
        patience = 3
        counter = 0

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            valid_loss = self.validate()
            print(
                f"Epoch [{epoch + 1}/{self.epochs}] | "
                f"Train Loss: {train_loss:.5f} | "
                f"Valid Loss: {valid_loss:.5f}"
            )

            if valid_loss < best_loss:
                best_loss = valid_loss
                counter = 0
                Path("checkpoints").mkdir(exist_ok=True)
                torch.save(self.model.state_dict(),"checkpoints/best_model.pth")

            else:
                counter += 1
                if counter >= patience:
                    print("\nEarly stopping triggered.")
                    break