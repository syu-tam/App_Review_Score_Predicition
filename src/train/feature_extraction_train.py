import torch
import torch.nn as nn
import os
from data.loaders import create_data_loader
from .optimizer import get_optimizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from models.model_builder import build_model_and_tokenizer
from utils.save_result import save_model_and_tokenizer, save_metrics_to_csv, save_plots, save_combined_config, create_unique_output_dir

def train_feature_extraction_model(review_column_name, dataframe, config):
    """
    特徴抽出モデルのトレーニングループ。メトリクスの記録、モデル・トークナイザー・設定の保存を行う。
    """
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer, model = build_model_and_tokenizer(config)  # モデルとトークナイザーの準備
    device = config.base.device
    model.to(device)

    # 設定関連
    num_epochs = config.feature_extraction.num_epochs
    all_train = config.feature_extraction.all_train

    # 出力ディレクトリの作成
    output_dir = create_unique_output_dir(config.base.base_output_dir, config.feature_extraction.output_dir)
    model_output_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_output_dir, exist_ok=True)

    # メトリクス記録用リスト
    epochs, train_losses, train_accuracies, train_qwks = [], [], [], []
    val_losses, val_accuracies, val_qwks = [], [], []

    # データローダーの準備
    if all_train:
        train_data_loader = create_data_loader(dataframe, tokenizer, review_column_name, config)
    else:
        train_data_loader, val_data_loader = create_data_loader(dataframe, tokenizer, review_column_name, config)

    # 最適化関数とスケジューラの設定
    optimizer = get_optimizer(model, config)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.feature_extraction.label_smoothing)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.feature_extraction.num_warmup_steps, 
                                                num_training_steps=len(train_data_loader) * num_epochs)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, correct_predictions, all_labels, all_preds = 0, 0, [], []

        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            review_input_ids, review_attention_mask = batch['review_input_ids'].to(device), batch['review_attention_mask'].to(device)
            reply_input_ids, reply_attention_mask = batch['reply_input_ids'].to(device), batch['reply_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # モデルのフォワードパスと損失計算
            outputs = model(review_input_ids=review_input_ids, review_attention_mask=review_attention_mask, 
                            reply_input_ids=reply_input_ids, reply_attention_mask=reply_attention_mask)

            loss = loss_fn(outputs, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        # エポックごとのトレーニングメトリクスを計算
        train_losses.append(total_train_loss / len(train_data_loader))
        train_accuracies.append((correct_predictions.double() / len(train_data_loader.dataset)).item())
        train_qwks.append(cohen_kappa_score(all_labels, all_preds, weights='quadratic'))
        epochs.append(epoch + 1)

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_losses[-1]} - Accuracy: {train_accuracies[-1]} - QWK: {train_qwks[-1]}")

        if not all_train:
            # 検証ループ
            model.eval()
            total_val_loss, correct_val_predictions, val_labels, val_preds = 0, 0, [], []

            with torch.no_grad():
                for batch in tqdm(val_data_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                    review_input_ids, review_attention_mask = batch['review_input_ids'].to(device), batch['review_attention_mask'].to(device)
                    reply_input_ids, reply_attention_mask = batch['reply_input_ids'].to(device), batch['reply_attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(review_input_ids=review_input_ids, review_attention_mask=review_attention_mask, 
                                    reply_input_ids=reply_input_ids, reply_attention_mask=reply_attention_mask)

                    loss = loss_fn(outputs, labels)
                    total_val_loss += loss.item()

                    _, preds = torch.max(outputs, dim=1)
                    correct_val_predictions += torch.sum(preds == labels)
                    val_labels.extend(labels.cpu().numpy())
                    val_preds.extend(preds.cpu().numpy())

            # 検証メトリクスの計算
            val_losses.append(total_val_loss / len(val_data_loader))
            val_accuracies.append((correct_val_predictions.double() / len(val_data_loader.dataset)).item())
            val_qwks.append(cohen_kappa_score(val_labels, val_preds, weights='quadratic'))

            print(f"Validation Loss: {val_losses[-1]} - Validation Accuracy: {val_accuracies[-1]} - Validation QWK: {val_qwks[-1]}")

    # モデルとトークナイザーの保存
    save_model_and_tokenizer(model, tokenizer, model_output_dir)

    # 使用したconfigファイルの保存
    save_combined_config(config, output_dir)

    if config.feature_extraction.save_metrics:
        # メトリクスの保存
        save_metrics_to_csv(epochs, train_losses, train_accuracies, train_qwks, val_losses, val_accuracies, val_qwks, output_dir, all_train)
        # メトリクスのプロットを保存
        save_plots(train_losses, train_accuracies, train_qwks, val_losses, val_accuracies, val_qwks, output_dir)

    return model, tokenizer
