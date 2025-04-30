import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange
import os

# Set both CPU and CUDA seeds
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class ImputationExperiment:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs('outputs', exist_ok=True)

    def torch_avg(self, tensor, weights):
        assert tensor.shape == weights.shape
        return torch.sum(tensor * weights) / torch.sum(weights)

    def expand_mask(self, mask):
        mask_expanded = []
        for col_idx, col in enumerate(self.q_df.columns):
            n_categories = len(self.q_df[col].cat.categories)
            mask_repeated = np.repeat(mask[:, col_idx][:, np.newaxis], n_categories, axis=1)
            mask_expanded.append(mask_repeated)
        mask_expanded = np.hstack(mask_expanded)
        return mask_expanded

    def expand_loss_weights(self, weights):
        weights_expanded = []
        for col_idx, col in enumerate(self.q_df.columns):
            n_categories = len(self.q_df[col].cat.categories)
            weights_expanded += ([weights[col_idx]] * n_categories)
        return weights_expanded

    def get_mode(self, series):
        return series.mode().iloc[0]

    @torch.compile
    def split_and_softmax(self, X):
        split_sizes = [self.border_indices[0]] + \
                    [self.border_indices[i+1] - self.border_indices[i] for i in range(len(self.border_indices)-1)] + \
                    [self.q_df.shape[1] - self.border_indices[-1]]

        split_sizes = [item * (i+2) for i, item in enumerate(split_sizes)]
        split_tensors = torch.split(X, split_sizes, dim=1)

        processed_tensors = []
        for i, tensor in enumerate(split_tensors):
            options_per_question = 2 + i
            n_questions = tensor.shape[1] // options_per_question
            reshaped = tensor.view(-1, n_questions, options_per_question)
            softmaxed = F.softmax(reshaped, dim=-1)
            processed_tensors.append(softmaxed)
        
        flattened_tensors = [tensor.view(tensor.shape[0], -1) for tensor in processed_tensors]
        X_softmaxed = torch.cat(flattened_tensors, dim=1)
        return X_softmaxed

    def get_test_loss(self, X_hat):
        X_imputed = X_hat
        imputed_test_values = X_imputed[self.test_mask_tensor]
        mse_low_rank = self.torch_avg(((self.test_values_tensor - imputed_test_values) ** 2).ravel(), self.loss_weights_test_tensor)
        return mse_low_rank.item()

    def load_data(self):
        self.raw_df = pd.read_feather("../data/data.feather")
        self.test_items_df = pd.read_csv("../data/test_items.csv", index_col=0)
        self.question_data = pd.read_csv("../data/question_data.csv", sep=';', index_col=0)
        self.question_weights = pd.read_csv('../outputs/question_weights.csv', index_col=0)
        self.q_weights = self.question_weights.scores.to_list()

        test_item_qs = [item for item in self.test_items_df.index if item in self.raw_df.columns]
        self.q_df = self.raw_df.drop(columns=test_item_qs)
        self.q_df = self.q_df.filter(regex=r'^q\d+$')

        N_DROP = 2000
        low_response = [col for col in self.question_data.sort_values('N').iloc[:N_DROP].index if col in self.q_df.columns]
        self.q_df = self.q_df.drop(columns=low_response)

        sorted_num_levels = self.q_df.apply(lambda x: len(x.cat.categories)).sort_values()
        self.q_df = self.q_df[sorted_num_levels.index]

        num_levels = self.q_df.nunique()
        sorted_num_levels = num_levels.sort_values()
        level_counts = sorted_num_levels.values
        diff = level_counts[1:] != level_counts[:-1]
        self.border_indices = list(diff.nonzero()[0] + 1)

    def transform_and_drop(self, df):
        X = self.preprocessor.transform(df)
        feature_names = self.preprocessor.get_feature_names_out()
        cols_to_keep = [i for i, name in enumerate(feature_names) if not name.endswith('_nan')]
        X = X[:, cols_to_keep]
        return X

    def prepare_data(self):
        non_nan = ~self.q_df.isna()
        non_nan_indices = np.flatnonzero(non_nan.values)

        np.random.seed(0)
        TEST_SIZE = 0.2
        test_size = int(len(non_nan_indices) * TEST_SIZE)
        test_mask_flat = np.random.choice(non_nan_indices, size=test_size, replace=False)

        self.test_mask = np.zeros_like(self.q_df.values, dtype=bool)
        self.test_mask.flat[test_mask_flat] = True

        self.df_masked = self.q_df.mask(self.test_mask)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(sparse_output=True, handle_unknown='ignore'), self.q_df.columns),
            ],
            sparse_threshold=1.0
        )

        self.preprocessor.fit(self.q_df)  # Fit the preprocessor here

        self.X_combined = self.transform_and_drop(self.df_masked)


        self.X_combined = self.transform_and_drop(self.df_masked)

        self.test_mask_expanded = self.expand_mask(self.test_mask)
        self.test_mask_sparse = sparse.csr_matrix(self.test_mask_expanded)
        self.original_mask = self.expand_mask(non_nan.values)

        self.weights_expanded = self.expand_loss_weights(self.q_weights)
        _, test_mask_qs = np.nonzero(self.test_mask_expanded)
        self.loss_weights_test = np.array(self.weights_expanded)[test_mask_qs]

        self.pr_q_answered = self.df_masked.notna().mean()
        self.pr_user_answered = self.df_masked.notna().mean(axis=1)

        self.feature_names = self.preprocessor.get_feature_names_out()
        self.kept_features = [item[8:].split('_') for item in self.feature_names if not item.endswith('_nan')]

        self.questions = np.array([q for q, o in self.kept_features])
        self.options   = np.array([o for q, o in self.kept_features])

        self.option_probs = {
            q: self.df_masked[q].value_counts(normalize=True).reindex(self.q_df[q].cat.categories, fill_value=0)
            for q in self.q_df.columns
        }
        self.option_probs_df = pd.DataFrame(self.option_probs).T

        self.users_idx, self.features_idx = self.test_mask_sparse.nonzero()

        self.q_for_masked = self.questions[self.features_idx]
        self.o_for_masked = self.options[self.features_idx]

        self.pr_user_vals = self.pr_user_answered.values[self.users_idx]
        self.pr_q_vals    = self.pr_q_answered[self.q_for_masked].values
        self.pr_option_vals = self.option_probs_df.values[
            self.option_probs_df.index.get_indexer(self.q_for_masked),
            self.option_probs_df.columns.get_indexer(self.o_for_masked)
        ]

        self.X_not_masked = self.transform_and_drop(self.q_df)

        self.row_mean = self.X_combined.mean(axis=1).ravel()
        self.col_mean = self.X_combined.mean(axis=0).ravel()

        self.X_combined_tensor = torch.tensor(self.X_combined.toarray(), dtype=torch.float32).to(self.device)
        self.test_mask_tensor = torch.tensor(self.test_mask_sparse.toarray(), dtype=torch.bool).to(self.device)
        self.train_mask_tensor = torch.tensor(self.original_mask, dtype=torch.bool).to(self.device) & ~self.test_mask_tensor

        self.train_mask_qs = torch.nonzero(self.train_mask_tensor)[:,1]
        self.loss_weights_train = torch.tensor(self.weights_expanded)[self.train_mask_qs.cpu()].to(self.device)
        self.loss_weights_test_tensor = torch.from_numpy(self.loss_weights_test).to(self.device)

        self.test_values_tensor = torch.from_numpy(self.X_not_masked[self.test_mask_sparse]).to(self.device)

    def naive_bayes_imputation(self):
        naive_imputed_values = self.pr_user_vals * self.pr_q_vals * self.pr_option_vals
        mse_naive = np.average(np.asarray(self.X_not_masked[self.test_mask_sparse] - naive_imputed_values).ravel()**2, weights=self.loss_weights_test)
        print(f"Naive Bayes MSE: {mse_naive}")
        return mse_naive

    def random_imputation(self):
        mse_random = np.average(np.asarray(self.X_not_masked[self.test_mask_sparse] - np.random.uniform(size=self.pr_user_vals.shape)).ravel() **2, weights=self.loss_weights_test)
        print(f"Random Imputation MSE: {mse_random}")
        return mse_random

    def constant_imputation(self):
        train_data = self.X_not_masked[self.original_mask & ~self.test_mask_sparse.toarray()]
        mean_val = train_data.mean()
        mse_constant = np.average(np.asarray(self.X_not_masked[self.test_mask_sparse] - (np.zeros_like(self.pr_user_vals) + mean_val)).ravel() **2, weights=self.loss_weights_test)
        print(f"Constant Imputation MSE: {mse_constant}")
        return mse_constant

    def modal_imputation(self):
        df_modal_imputed = self.df_masked.apply(lambda col: col.fillna(self.get_mode(col)))
        X_modal = self.transform_and_drop(df_modal_imputed)
        mse_modal = np.average(np.asarray(self.X_not_masked[self.test_mask_sparse] - X_modal[self.test_mask_sparse]).ravel() **2, weights=self.loss_weights_test)
        print(f"Modal Imputation MSE: {mse_modal}")
        return mse_modal

    def low_rank_approximation(self):
        rank = 5
        learning_rate = 0.2
        epochs = 50

        n_users, n_features = self.X_combined.shape
        B = torch.randn(n_users, rank, requires_grad=True, device=self.device)
        C = torch.randn(rank, n_features, requires_grad=True, device=self.device)
        X_hat = torch.mm(B, C)

        optimizer = optim.Adam([B, C], lr=learning_rate)

        train_losses = []
        test_losses = []
        for epoch in trange(epochs):
            optimizer.zero_grad()
            X_hat = torch.mm(B, C)
            X_hat = self.split_and_softmax(X_hat)
            loss = self.torch_avg((X_hat[self.train_mask_tensor] - self.X_combined_tensor[self.train_mask_tensor]) ** 2, weights=self.loss_weights_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            test_losses.append(self.get_test_loss(X_hat))

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (weighted MSE)')
        plt.title('Low Rank Approximation Training')
        plt.legend()
        plt.savefig('outputs/lora_loss.png')
        plt.close()

        mse_low_rank = self.get_test_loss(X_hat)
        print(f"Low-rank MSE: {mse_low_rank}")

        return mse_low_rank, X_hat

    def autoencoder_imputation(self):
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, encoding_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, input_dim)
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        input_dim = self.X_combined_tensor.shape[1]
        encoding_dim = 5
        learning_rate = 0.001
        epochs = 50

        model = Autoencoder(input_dim, encoding_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses_ae = []
        test_losses_ae = []
        for epoch in trange(epochs):
            optimizer.zero_grad()
            outputs = model(self.X_combined_tensor)
            outputs_softmax = self.split_and_softmax(outputs)
            loss = self.torch_avg((outputs_softmax[self.train_mask_tensor] - self.X_combined_tensor[self.train_mask_tensor]) ** 2, weights=self.loss_weights_train)
            loss.backward()
            optimizer.step()
            test_loss = self.get_test_loss(outputs_softmax)
            train_losses_ae.append(loss.item())
            test_losses_ae.append(test_loss)

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_ae, label='Train Loss')
        plt.plot(test_losses_ae, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (weighted MSE)')
        plt.title('Autoencoder Training')
        plt.legend()
        plt.savefig('outputs/ae_loss.png')
        plt.close()

        model.eval()
        with torch.no_grad():
            imputed_data = model(self.X_combined_tensor)
            imputed_data_softmax = self.split_and_softmax(imputed_data)

        mse_autoencoder = self.get_test_loss(imputed_data_softmax)
        print(f"Autoencoder MSE: {mse_autoencoder}")

        return mse_autoencoder

    def compare_methods(self, errors):
        e_series = pd.Series(errors).sort_values()
        plt.figure(figsize=(10,6))
        sns.barplot(x=e_series.index, y=e_series.values, palette='viridis')
        plt.title('Weighted Squared Error by Imputation Method')
        plt.xlabel('Imputation Method')
        plt.ylabel('Weighted Squared Error')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/imputers.png')
        plt.close()

    def save_imputed_data(self, X_hat):
        imputed_onehot = X_hat.detach().cpu().numpy()
        np.save('outputs/imputed_lora.npy', imputed_onehot)

        original = self.X_combined.toarray().copy()
        original[~self.original_mask] = imputed_onehot[~self.original_mask]
        np.save('outputs/imputed_with_original.npy', original)

        with_indicators = np.concatenate([imputed_onehot, self.q_df.isna().values], axis=1)
        np.save('outputs/imputed_lora_indicators.npy', with_indicators)

        with_indicators_original = np.concatenate([original, self.q_df.isna().values], axis=1)
        np.save('outputs/imputed_lora_indicators_original.npy', with_indicators_original)

    def analyze_feature_importance(self, C):
        feature_importance = (C.detach() ** 2).sum(dim=0).cpu().numpy()
        feature_importance /= feature_importance.sum()

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.xlabel('Feature Index')
        plt.ylabel('Normalized Importance')
        plt.title('Feature Importances from Low-Rank Approximation')
        plt.savefig('outputs/feature_importance.png')
        plt.close()

        pairs = []
        for col in self.q_df.columns:
            cats = self.q_df[col].cat.categories
            pairs.extend((col, cat) for cat in cats)

        triples = []
        for pair, importance in zip(pairs, feature_importance.tolist()):
            triples.append((pair[0], pair[1], importance))

        triples.sort(reverse=True, key=lambda x: x[2])

        print("Top 10 important features:")
        for q, option, importance in triples[:10]:
            print(f"{q}, {self.question_data.loc[q].text}, Option: {option}, Importance: {importance:.3f}")

    def get_entropies(self, X_hat):
        split_sizes = [self.border_indices[0]] + \
                    [self.border_indices[i+1] - self.border_indices[i] for i in range(len(self.border_indices)-1)] + \
                    [self.q_df.shape[1] - self.border_indices[-1]]
        split_sizes = [item * (i+2) for i, item in enumerate(split_sizes)]
        split_tensors = torch.split(X_hat, split_sizes, dim=1)
        
        normalized_entropies = []
        for i, tensor in enumerate(split_tensors):
            options = 2 + i
            max_entropy = torch.log(torch.tensor(options))
            reshaped = tensor.view(-1, tensor.shape[1]//options, options)
            entropy = -torch.sum(reshaped * torch.log(reshaped + 1e-10), dim=-1)
            normalized_entropies.append(entropy / max_entropy)
        
        return torch.cat(normalized_entropies, dim=1)

    def analyze_entropies(self, X_hat):
        entropies = self.get_entropies(X_hat).flatten().detach().cpu()
        missing_entropies = pd.Series(entropies[self.q_df.isna().values.flatten()])
        plt.figure(figsize=(10, 6))
        missing_entropies.hist()
        plt.title('Distribution of Entropies for Missing Values')
        plt.xlabel('Normalized Entropy')
        plt.ylabel('Frequency')
        plt.savefig('outputs/entropy_distribution.png')
        plt.close()

    def run_experiments(self):
        print("loading data...")
        self.load_data()
        print("preparing data...")
        self.prepare_data()

        print("running naive...")
        mse_naive = self.naive_bayes_imputation()
        print("running random...")
        mse_random = self.random_imputation()
        print("running constant...")
        mse_constant = self.constant_imputation()
        print("running model...")
        mse_modal = self.modal_imputation()
        print("running low rank...")
        mse_low_rank, X_hat = self.low_rank_approximation()
        print("running autoencoder...")
        mse_autoencoder = self.autoencoder_imputation()


        errors = {
            "Autoencoder": mse_autoencoder,
            "Constant": mse_constant,
            "Low Rank": mse_low_rank,
            "Modal": mse_modal,
            "Naive": mse_naive,
            "Random": mse_random
        }

        self.compare_methods(errors)
        self.save_imputed_data(X_hat)
        self.analyze_feature_importance(X_hat)
        self.analyze_entropies(X_hat)

        print("All experiments completed. Results and plots saved in the 'outputs' directory.")

if __name__ == "__main__":
    experiment = ImputationExperiment()
    experiment.run_experiments()