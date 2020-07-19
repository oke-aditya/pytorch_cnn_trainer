# Just an example of how quantization aware training can be used with the same engine.

import timm
import config
from pytorch_cnn_trainer import model_factory
from pytorch_cnn_trainer import engine
from tqdm import tqdm
import torch.nn as nn
import utils
from pytorch_cnn_trainer import dataset
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import torch.optim as optim
from torch.quantization import convert

if __name__ == "__main__":

    train_ds, valid_ds = dataset.create_cifar10_dataset(
        config.train_transforms, config.valid_transforms
    )
    train_loader, valid_loader = dataset.create_loaders(train_ds, valid_ds)

    qat_model = model_factory.create_timm_model(
        config.MODEL_NAME, config.NUM_ClASSES, config.IN_CHANNELS, config.PRETRAINED
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(qat_model.parameters(), lr=1e-3)
    early_stopper = utils.EarlyStopping(patience=3, verbose=True, path=config.SAVE_PATH)

    qat_model.config = torch.quantization.get_default_qat_qconfig("fbgemm")
    _ = torch.quantization.prepare_qat(qat_model, inplace=True)

    # We can fine-tune / train the qat_models on GPU too.

    for param in qat_model.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = qat_model.to(device)

    # NUM_TRAIN_BATCHES = 5 You can pass these too in train step if you want small subset to train
    # NUM_VAL_BATCHES = 5  You can pass these too in train step if you want small subset to validate

    history = engine.fit(
        epochs=config.EPOCHS,
        model=qat_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        early_stopper=early_stopper,
    )

    qat_model.cpu()  # We need to move to cpu for conversion.

    qat_model_trained = convert(qat_model, inplace=False)
    print("Converted the Quantization aware training model.")
    # torch.save(model_quantized_and_trained.state_dict(), config.QAT_SAVE_PATH)
