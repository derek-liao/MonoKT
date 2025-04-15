import os
import argparse
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import yaml
import wandb
from data_loaders import (
    MostRecentQuestionSkillDataset,
    MostEarlyQuestionSkillDataset,
    MostRecentQuestionSkillStackedDataset,
    MostEarlyQuestionSkillStackedDataset,
    SimCLRDatasetWrapper,
    CounterDatasetWrapper,
)
from models.cl4kt import CL4KT
from models.dkt import DKT
from models.sakt import SAKT
from models.dkvmn import DKVMN
from models.simplekt import simpleKT
from models.diskt import DisKT
from models.akt import AKT
from models.corekt import CoreKT
from models.dtransformer import DTransformer
from models.atkt import ATKT
from models.folibikt import folibiKT
from models.skvmn import SKVMN
from models.deep_irt import DeepIRT
from models.sparsekt import sparseKT
from models.gkt import GKT
from models.gkt_utils import get_gkt_graph
from models.mikt import MIKT
from models.routerkt import RouterKT
from models.qikt_moe import QIKTMOE
from models.atdkt import ATDKT
# from models.mamba4kt import Mamba4KT
from train import model_train
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from utils.config import ConfigNode as CN
from utils.file_io import PathManager


def main(config):
    accelerator = Accelerator()
    device = accelerator.device

    # print 可用的GPU个数
    print(f"Available GPUs: {torch.cuda.device_count()}")

    model_name = config.model_name
    dataset_path = config.dataset_path
    data_name = config.data_name
    seed = config.seed
    test_name = config.test_name

    # Initialize wandb if it's not already initialized (direct run from main.py)
    if config.train_config.log_wandb:
        wandb.init(
            project="kt-experiment",
            config={
                "model_name": model_name,
                "data_name": data_name,
                "seed": seed,
                "test_name": test_name,
                "sequence_option": config.train_config.sequence_option,
                "batch_size": config.train_config.batch_size,
                "learning_rate": config.train_config.learning_rate,
                "optimizer": config.train_config.optimizer,
                "seq_len": config.train_config.seq_len,
            }
        )
        # Add model specific config to wandb
        if hasattr(config, f"{model_name}_config"):
            model_config_dict = dict(getattr(config, f"{model_name}_config"))
            for key, value in model_config_dict.items():
                wandb.config.update({key: value})

    np.random.seed(seed)
    torch.manual_seed(seed)


    train_config = config.train_config
    checkpoint_dir = config.checkpoint_dir

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ckpt_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, data_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = train_config.batch_size
    eval_batch_size = train_config.eval_batch_size
    learning_rate = train_config.learning_rate
    optimizer = train_config.optimizer
    seq_len = train_config.seq_len

    if train_config.sequence_option == "recent":  # the most recent N interactions
        print("Using recent dataset")
        dataset = MostRecentQuestionSkillDataset
    elif train_config.sequence_option == "early":  # the most early N interactions
        print("Using early dataset")
        dataset = MostEarlyQuestionSkillDataset
    elif train_config.sequence_option == "recent_stack":
        print("Using recent_stack dataset")
        dataset = MostRecentQuestionSkillStackedDataset
    elif train_config.sequence_option == "early_stack":  # the most early N interactions with stacking
        print("Using early_stack dataset")
        dataset = MostEarlyQuestionSkillStackedDataset
    else:
        raise NotImplementedError("sequence option is not valid")

    test_aucs, test_accs, test_rmses = [], [], []

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    df_path = os.path.join(os.path.join(dataset_path, data_name), "preprocessed_df.csv")
    df = pd.read_csv(df_path, sep="\t")

    print("skill_min", df["skill_id"].min())
    users = df["user_id"].unique()
    df["skill_id"] += 1  # zero for padding
    df["item_id"] += 1  # zero for padding
    num_skills = df["skill_id"].max() + 1
    num_questions = df["item_id"].max() + 1

    np.random.shuffle(users)

    print("MODEL", model_name)
    print(dataset)
    if data_name in ["statics", "assistments15"]:
        num_questions = 0

    for fold, (train_ids, test_ids) in enumerate(kfold.split(users)):
        if model_name == "cl4kt":
            model_config = config.cl4kt_config
            model = CL4KT(num_skills, num_questions, seq_len, **model_config)
            mask_prob = model_config.mask_prob
            crop_prob = model_config.crop_prob
            permute_prob = model_config.permute_prob
            replace_prob = model_config.replace_prob
            negative_prob = model_config.negative_prob
        elif model_name == 'dkt':
            model_config = config.dkt_config
            model = DKT(num_skills, **model_config)
        elif model_name == 'sakt':
            model_config = config.sakt_config
            model = SAKT(num_skills, seq_len, **model_config)
        elif model_name == 'dkvmn':
            model_config = config.dkvmn_config
            model = DKVMN(num_skills, **model_config)
        elif model_name == 'skvmn':
            model_config = config.skvmn_config
            model = SKVMN(num_skills, **model_config)
        elif model_name == 'deep_irt':
            model_config = config.deep_irt_config
            model = DeepIRT(num_skills, **model_config)
        elif model_name == 'simplekt':
            model_config = config.simplekt_config
            model = simpleKT(num_skills, num_questions, seq_len, **model_config)
        elif model_name == 'diskt':
            model_config = config.diskt_config
            model = DisKT(num_skills, num_questions, seq_len, **model_config)
        elif model_name == "akt":
            model_config = config.akt_config
            if data_name in ["statics", "assistments15"]:
                num_questions = 0
            model = AKT(num_skills, num_questions, **model_config)
        elif model_name == 'atkt':
            model_config = config.atkt_config
            model = ATKT(num_skills, **model_config)
        elif model_name == 'atdkt':
            model_config = config.atdkt_config
            model = ATDKT(num_skills, num_questions, seq_len, **model_config)
        elif model_name == 'folibikt':
            model_config = config.folibikt_config
            model = folibiKT(num_skills, num_questions, seq_len, **model_config)
        elif model_name == "sparsekt":
            model_config = config.sparsekt_config
            model = sparseKT(num_skills, num_questions, seq_len, **model_config)
        elif model_name == 'gkt':
            model_config = config.gkt_config
            graph_type = model_config['graph_type']
            fname = f"gkt_graph_{graph_type}.npz"
            graph_path = os.path.join(os.path.join(dataset_path, data_name), fname)
            if os.path.exists(graph_path):
                graph = torch.tensor(np.load(graph_path, allow_pickle=True)['matrix']).float()
            else:
                graph = get_gkt_graph(df, num_skills, graph_path, graph_type=graph_type)
                graph = torch.tensor(graph).float()
            model = GKT(device, num_skills, graph, **model_config)
        elif model_name == "corekt":
            model_config = config.corekt_config
            if data_name in ["statics", "assistments15"]:
                num_questions = 0
            print(f'model_config: {model_config}')
            model = CoreKT(num_skills, num_questions, **model_config)
        elif model_name == 'dtransformer':
            model_config = config.dtransformer_config
            if data_name in ["statics", "assistments15"]:
                num_questions = 0
            model = DTransformer(num_skills, num_questions, **model_config)
        elif model_name == 'mikt':
            model_config = config.mikt_config
            pro2skill = torch.zeros((num_questions, num_skills)).to(device)
            for (x, y) in zip(df["item_id"].tolist(), df["skill_id"].tolist()):
                pro2skill[x][y]=1
            model = MIKT(num_skills, num_questions, seq_len, pro2skill, **model_config)
        # elif model_name == 'mamba4kt':
        #     model_config = config.mamba4kt_config
        #     model = Mamba4KT(num_skills, num_questions, **model_config)
        elif model_name == 'routerkt':
            model_config = config.routerkt_config
            model = RouterKT(num_skills, num_questions, seq_len, **model_config)
        elif model_name == 'qiktmoe':
            model_config = config.qikt_moe_config
            model = QIKTMOE(num_skills, num_questions, seq_len, **model_config)

        train_users = users[train_ids]
        np.random.shuffle(train_users)
        offset = int(len(train_ids) * 0.9)

        valid_users = train_users[offset:]
        train_users = train_users[:offset]

        test_users = users[test_ids]

        train_df = df[df["user_id"].isin(train_users)]
        valid_df = df[df["user_id"].isin(valid_users)]
        test_df = df[df["user_id"].isin(test_users)]


        train_dataset = dataset(train_df, seq_len, num_skills, num_questions)
        valid_dataset = dataset(valid_df, seq_len, num_skills, num_questions)
        test_dataset = dataset(test_df, seq_len, num_skills, num_questions)

        print("train_ids", len(train_users))
        print("valid_ids", len(valid_users))
        print("test_ids", len(test_users))

        if "cl" in model_name:  # contrastive learning
            train_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        train_dataset,
                        seq_len,
                        mask_prob,
                        crop_prob,
                        permute_prob,
                        replace_prob,
                        negative_prob,
                        eval_mode=False,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        valid_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        test_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )
        elif "dis" in model_name:  # diskt
            train_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        train_dataset,
                        seq_len,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        valid_dataset,
                        seq_len,
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        test_dataset,
                        seq_len,
                    ),
                    batch_size=eval_batch_size,
                )
            )
        else:
            train_loader = accelerator.prepare(
                DataLoader(train_dataset, batch_size=batch_size)
            )

            valid_loader = accelerator.prepare(
                DataLoader(valid_dataset, batch_size=eval_batch_size)
            )

            test_loader = accelerator.prepare(
                DataLoader(test_dataset, batch_size=eval_batch_size)
            )

        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate, weight_decay=train_config.wl)

        model, opt = accelerator.prepare(model, opt)


        test_auc, test_acc, test_rmse = model_train(
            fold,
            model,
            accelerator,
            opt,
            train_loader,
            valid_loader,
            test_loader,
            config,
            n_gpu,
        )


        test_aucs.append(test_auc)
        test_accs.append(test_acc)
        test_rmses.append(test_rmse)
        
        # Log individual fold results to wandb only if log_wandb is True
        if config.train_config.log_wandb:
            wandb.log({
                f"fold_{fold}/test_auc": test_auc,
                f"fold_{fold}/test_acc": test_acc,
                f"fold_{fold}/test_rmse": test_rmse
            })

    test_auc = np.mean(test_aucs)
    test_auc_std = np.std(test_aucs)
    test_acc = np.mean(test_accs)
    test_acc_std = np.std(test_accs)
    test_rmse = np.mean(test_rmses)
    test_rmse_std = np.std(test_rmses)

    now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")  # KST time

    # Log final results to wandb
    if config.train_config.log_wandb:
        wandb.log({
            "test_auc": test_auc,
            "test_acc": test_acc,
            "test_rmse": test_rmse,
            "test_auc_std": test_auc_std,
            "test_acc_std": test_acc_std,
            "test_rmse_std": test_rmse_std,
            "test_aucs": test_aucs,
            "test_accs": test_accs,
            "test_rmses": test_rmses
        })

    print("\n5-fold CV Result")
    print("AUC\tACC\tRMSE")
    print("{:.5f}\t{:.5f}\t{:.5f}".format(test_auc, test_acc, test_rmse))

    return test_auc, test_acc, test_rmse


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="diskt",
        help="The name of the model to train. \
            The possible models are in [akt, cl4kt, dkt, sakt, simplekt, dkvmn...]. \
            The default model is diskt.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="spanish",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--reg_cl",
        type=float,
        default=0.1,
        help="regularization parameter contrastive learning loss",
    )
    parser.add_argument("--mask_prob", type=float, default=0.2, help="mask probability")
    parser.add_argument("--crop_prob", type=float, default=0.3, help="crop probability")
    parser.add_argument(
        "--permute_prob", type=float, default=0.3, help="permute probability"
    )
    parser.add_argument(
        "--replace_prob", type=float, default=0.3, help="replace probability"
    )
    parser.add_argument(
        "--negative_prob",
        type=float,
        default=1.0,
        help="reverse responses probability for hard negative pairs",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout probability"
    )
    parser.add_argument(
        "--batch_size", type=float, default=512, help="train batch size"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=64, help="embedding size"
    )
    parser.add_argument(
        "--state_d", type=int, default=64, help="hidden size"
    )
    parser.add_argument(
        "--test_name", type=str, default='low', help="the possible testsets are in [ednet-low, ednet-medium, ednet-high]"
    )
    parser.add_argument("--l2", type=float, default=1e-5, help="l2 regularization param")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    args = parser.parse_args()

    base_cfg_file = PathManager.open("configs/example.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    cfg.test_name = args.test_name
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.eval_batch_size = int(args.batch_size)
    cfg.train_config.learning_rate = args.lr
    cfg.train_config.optimizer = args.optimizer

    if args.model_name == "cl4kt":
        cfg.cl4kt_config.reg_cl = args.reg_cl
        cfg.cl4kt_config.mask_prob = args.mask_prob
        cfg.cl4kt_config.crop_prob = args.crop_prob
        cfg.cl4kt_config.permute_prob = args.permute_prob
        cfg.cl4kt_config.replace_prob = args.replace_prob
        cfg.cl4kt_config.negative_prob = args.negative_prob
        cfg.cl4kt_config.dropout = args.dropout
        cfg.cl4kt_config.l2 = args.l2
    elif args.model_name == 'dkt':  # dkt
        cfg.dkt_config.dropout = args.dropout
    elif args.model_name == 'sakt':  # sakt
        cfg.sakt_config.dropout = args.dropout
    elif args.model_name == 'dkvmn':  # dkvmn
        cfg.dkvmn_config.dropout = args.dropout
    elif args.model_name == 'skvmn':  # skvmn
        cfg.skvmn_config.dropout = args.dropout
    elif args.model_name == 'deep_irt':  # deep_irt
        cfg.deep_irt_config.dropout = args.dropout
    elif args.model_name == 'simplekt':  # simplekt
        cfg.simplekt_config.dropout = args.dropout
    elif args.model_name == 'diskt':  # dikt
        cfg.diskt_config.dropout = args.dropout
    elif args.model_name == 'akt':  # akt
        cfg.akt_config.l2 = args.l2
        cfg.akt_config.dropout = args.dropout
    elif args.model_name == 'atkt':  # atkt
        cfg.atkt_config.dropout = args.dropout
    elif args.model_name == 'atdkt':  # atdkt
        cfg.atdkt_config.dropout = args.dropout
    elif args.model_name == 'folibikt':  # folibikt
        cfg.folibikt_config.l2 = args.l2
        cfg.folibikt_config.dropout = args.dropout
    elif args.model_name == 'sparsekt':  # sparsekt
        cfg.sparsekt_config.dropout = args.dropout
    elif args.model_name == 'gkt':  # gkt
        cfg.gkt_config.dropout = args.dropout
    elif args.model_name == 'corekt':  # corekt
        cfg.corekt_config.l2 = args.l2
        cfg.corekt_config.dropout = args.dropout
    elif args.model_name == 'dtransformer':  # dtransformer
        cfg.dtransformer_config.dropout = args.dropout
        cfg.dtransformer_config.embedding_size = args.embedding_size
    elif args.model_name == 'mikt':  # mikt
        cfg.mikt_config.state_d = args.state_d
        cfg.mikt_config.dropout = args.dropout
        cfg.mikt_config.embedding_size = args.embedding_size
    elif args.model_name == 'routerkt':  # routerkt
        cfg.routerkt_config.dropout = args.dropout
    elif args.model_name == 'qiktmoe':  # qiktmoe
        cfg.qikt_moe_config.dropout = args.dropout
        cfg.qikt_moe_config.embedding_size = args.embedding_size
        cfg.qikt_moe_config.l2 = args.l2
    # elif args.model_name == 'mamba4kt':  # mamba4kt
    #     cfg.mamba4kt_config.dropout = args.dropout

    cfg.freeze()

    # print(cfg)
    main(cfg)
