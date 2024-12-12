# @Time    : 2024/06/01 21:09
# @Author  : Yang Xionghui
# @File    : train.py
# @Software: PyCharm
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torchvision.models
from torchvision.transforms import transforms
from timm.loss import LabelSmoothingCrossEntropy,SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import cohen_kappa_score
import time
import torch
torch.backends.cudnn.enabled = False
import sys
import argparse
from timm.utils import *
from test import ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from torch.utils.data.distributed import DistributedSampler

from HCGN import HCGN_CONVNEXT_TINY, HCGN_VGG11, HCGN_VGG13, HCGN_VGG16, HCGN_VGG19, HCGN_RESNET18, HCGN_RESNET34, HCGN_RESNET50, HCGN_RESNET101, HCGN_RESNET152

def convert_arg_line_to_args(arg_line):
    return arg_line.split()
parser = argparse.ArgumentParser(description="CGNN_OAI", fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args
parser.add_argument('--model_name', type=str, help="the model name",default='HCGN_CONVNEXT_TINY')
parser.add_argument('--pretrained', action='store_true', help="if vgg pretrained",default=False)

# dataset
parser.add_argument('--data_path', type=str, help="the path of your train datasets",default=r'')

# training
parser.add_argument('--batch_size', type=int, help="batch size", default=16)
parser.add_argument('--total_steps', type=int, help='the total iteration number', default=500000)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
parser.add_argument('--cnn_lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--num_classes', type=int, help='num_classes', default=5)
parser.add_argument('--weight_decay', type=float, help='weight decay factor for optimization', default=1e-4)

# log and save
parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')
parser.add_argument('--log_directory', type=str, help='directory to save summaries',default=r'./logs/')
parser.add_argument('--log_name', type=str, help='name for log_directory', default='1')
parser.add_argument('--log_freq', type=int, help='Logging frequency in global steps', default=20)

# online eval
parser.add_argument('--do_online_eval', help='if set, perform online eval in every eval_freq steps',action='store_true',default=True)
parser.add_argument('--eval_freq', type=int, help='Online evaluation frequency in global steps', default=50)
parser.add_argument('--patience', type=int, help='patience times to adjust lr if eval acc can not be better',default=100)

#
parser.add_argument('--optim', help='the optimizer', type=str, default='sgd')
parser.add_argument('--Mixup', help='whether to use mixup', type=str,default=False)
parser.add_argument('--Ema', help='whether to use Ema', type=str,default=False)
parser.add_argument('--smoothing', type=str, default=False, help='Label smoothing (default: 0.1)')

#pretrained network

#vig network
parser.add_argument('--k', help='neighbor num', type=int, default=1)
parser.add_argument('--epsilon', help='stochastic epsilon for gcn', type=float, default=0.2)
parser.add_argument('--use_stochastic', help='stochastic for gcn true or false', default=False)
parser.add_argument('--heads', help='num heads for gat', type=int, default=1)

parser.add_argument('--img_size', help='input img size', type=int, default=224)

if sys.argv.__len__()==2:
    argsfile_with_prefix='@'+sys.argv[1]
    args=parser.parse_args([argsfile_with_prefix])
    print(args)
else:
    args=parser.parse_args()
    print(args)
class MyScheduler:
    def __init__(self, optimizer, min_lr):
        self.optimizer = optimizer
        self.min_lr = min_lr
    def update_lr(self, ):
        # if len(self.lr_groups) != 1:
        #     lr = self.lr_groups.pop(0)
        # else:
        #     lr = self.lr_groups[0]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

def train_model(args):
    global arg
    arg =args
    
    random_state = 21
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    torch.set_num_threads(2)
    torch.cuda.empty_cache()

    
    torch.distributed.init_process_group(backend='nccl',init_method='env://')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        print("total {} GPUs".format(torch.cuda.device_count()))

    writer = SummaryWriter(os.path.join(args.log_directory, args.log_name), flush_secs=30)
    # 将train文件复制到相应目录
    command= 'cp ' + './train.py' +' '+ os.path.join(args.log_directory, args.log_name)
    os.system(command)
    #将配置文件复制到相应目录
    command = 'cp ' + './run.txt' + ' ' + os.path.join(args.log_directory, args.log_name)
    os.system(command)
    # 将model文件复制到相应目录
    command = 'cp ' + './HCGN.py' + ' ' + os.path.join(args.log_directory, args.log_name)
    os.system(command)
    
    if local_rank == 0:
        print("total %d steps, batch_size %d" % (args.total_steps, args.batch_size))

    pixel_mean_train, pixel_std_train = 0.66133188,  0.21229856
    
    data_transform = {
                    "train": transforms.Compose([
                            transforms.ColorJitter(brightness=.33, saturation=.33),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomAffine(degrees=(-10, 10), scale=(0.9, 1.10)),
                            transforms.Resize((args.img_size,args.img_size)), 
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor(),
                            transforms.Normalize([pixel_mean_train] * 3, [pixel_std_train] * 3)]),
                    "test": transforms.Compose([
                                    transforms.Resize((args.img_size, args.img_size)), 
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    transforms.Normalize([pixel_mean_train] * 3, [pixel_std_train] * 3),
                                    ])}
 
    
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_path, "train"),transform=data_transform["train"])
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True, num_workers=args.num_workers)

    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_path, "test"),transform=data_transform["test"])
    test_sampler = DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers)

    if local_rank == 0:
        print('train:', len(train_dataset))
        print('test:',len(test_dataset))

    if args.model_name in ['HCGN_VGG11']:
        net = HCGN_VGG11(args=args)
    elif args.model_name in ['HCGN_VGG13']:
        net = HCGN_VGG13(args=args)
    elif args.model_name in ['HCGN_VGG16']:
        net = HCGN_VGG16(args=args)
    elif args.model_name in ['HCGN_VGG19']:
        net = HCGN_VGG19(args=args)
    elif args.model_name in ['HCGN_RESNET18']:
        net = HCGN_RESNET18(args=args)
    elif args.model_name in ['HCGN_RESNET34']:
        net = HCGN_RESNET34(args=args)
    elif args.model_name in ['HCGN_RESNET50']:
        net = HCGN_RESNET50(args=args)
    elif args.model_name in ['HCGN_RESNET101']:
        net = HCGN_RESNET101(args=args)
    elif args.model_name in ['HCGN_RESNET152']:
        net = HCGN_RESNET152(args=args)
    elif args.model_name in ['HCGN_CONVNEXT_TINY']:
        net = HCGN_CONVNEXT_TINY(args=args)
    else:
        print("No model named " + args.model_name)
        exit(1)
    
    if local_rank == 0:
        print(net)
    model=torch.nn.parallel.DistributedDataParallel(net.to(device))

    # for name, parms in model.named_parameters():
    #     print('-->name:', name)
    num_params = sum(p.numel() for p in model.parameters())
    

    num_params_update = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if local_rank == 0:
        print("== Total number of parameters: {:.2f}M".format(num_params / (1024 * 1024)))
        print("== Total number of learning parameters: {:.2f}M".format(num_params_update / (1024 * 1024)))
        print("== Model Initialized")
    writer.add_text('parameters', str(num_params_update))
    # 定义loss和optimizer
    model_ema = None
    if args.Ema:
        if local_rank == 0:
            print("USE EMA!")
        model_ema = ModelEma(model,decay=0.99996)

    mixup_fn = None
    if args.Mixup:
        if local_rank == 0:
            print("USE Mixup!")
        mixup_fn = Mixup(
                mixup_alpha=0.8, cutmix_alpha=1.0,cutmix_minmax=None,
                prob=1.0, switch_prob=0.5, mode='batch',
                label_smoothing=0.1,num_classes=5)
    if args.Mixup:
        # smoothing is handled with mixup label transform
        criterion_train = SoftTargetCrossEntropy().to(device)
    elif args.smoothing:
        if local_rank == 0:
            print("USE LabelSmoothing!")
        criterion_train = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
    else:
        criterion_train = torch.nn.CrossEntropyLoss().to(device)
    criterion_val = torch.nn.CrossEntropyLoss().to(device)

    if args.model_name in ["vgg16_original", "vgg19_original"]:
            params = [
            {"params": net.features.parameters(), "lr": args.cnn_lr},
            {"params": net.classifier.parameters(), "lr": args.lr},
        ]
    elif args.model_name in ["resnet50_original", "resnet101_original"]:
            params = [
            {"params": net.conv1.parameters(), "lr": args.cnn_lr},
            {"params": net.bn1.parameters(), "lr": args.cnn_lr},
            {"params": net.relu.parameters(), "lr": args.cnn_lr},
            {"params": net.maxpool.parameters(), "lr": args.cnn_lr},
            {"params": net.layer1.parameters(), "lr": args.cnn_lr},
            {"params": net.layer2.parameters(), "lr": args.cnn_lr},
            {"params": net.layer3.parameters(), "lr": args.cnn_lr},
            {"params": net.layer4.parameters(), "lr": args.cnn_lr},
            {"params": net.avgpool.parameters(), "lr": args.cnn_lr},
            {"params": net.fc.parameters(), "lr": args.lr},
        ]
    else:
        base_params = list(map(id, net.backbone.parameters()))
        other_params = filter(lambda p: id(p) not in base_params, model.parameters())

        params = [
            {"params": net.backbone.parameters(), "lr": args.cnn_lr},
            {"params": other_params, "lr": args.lr},
        ]

    if args.optim in ["sgd", "Sgd", "SGD"]:
        optimizer = torch.optim.SGD(params,  momentum=0.9, nesterov=True,weight_decay=args.weight_decay)
    elif args.optim in ["Adam", "adam", "ADAM"]:
        optimizer = torch.optim.Adam(params, weight_decay=args.weight_decay)
    elif args.optim in ["Adamw", "adamw", "ADAMW"]:
        optimizer = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    elif args.optim in ["RMSprop", "Rmsprop", "rmsprop"]:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,weight_decay=args.weight_decay, momentum=0.9)
    else:
        optimizer =None

    scheduler = MyScheduler(optimizer=optimizer, min_lr=0.0001)
    global_step = 0
    best_test_acc = 0
    best_test_step = 0
    patience = 0

    total_steps = args.total_steps
    start_time = time.time()
    duration = 0
    if local_rank == 0:
        print('*' * 10, 'start training', '*' * 10)
    model.train()
    while global_step < total_steps:
        train_loader.sampler.set_epoch(global_step)
        for _, data in enumerate(train_loader):
            optimizer.zero_grad()
            before_op_time = time.time()
            # inputs, labels = data
            inputs,label = data
            # print(labels)
            labels = torch.as_tensor(label)
            #inputs, labels = inputs.to(device), labels.to(device)
            inputs,labels = inputs.to(device), labels.to(device)
            # if mixup_fn is not None:
            #     inputs, labels = mixup_fn(inputs, labels)
            # outputs = model(inputs)
            outputs = model(inputs)
            loss = criterion_train(outputs, labels)
            loss.backward()

            loss = gather_tensors(loss).mean()

            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)
            pred = outputs.argmax(dim=1)
            # scheduler.step()

            # print("pred:",pred.shape)
            # print("labels:",labels.shape)
            train_correct = (pred == labels).sum().to(device)
            # train_correct = (outputs == labels).sum().to(device)
            train_acc = train_correct / labels.size(0)
            train_acc = gather_tensors(train_acc).mean()
            
            cnn_lr = optimizer.state_dict()['param_groups'][0]['lr']
            current_lr = optimizer.state_dict()['param_groups'][-1]['lr'] #gnn的lr

            if local_rank == 0:
                print(
                    '[gobal step/total steps]: [{}/{}], base lr: {:.8f},lr: {:.8f}, loss: {:.8f}, train acc:{:.8f}'.format(global_step,
                                                                                                        total_steps,
                                                                                                        cnn_lr,
                                                                                                        current_lr,
                                                                                                        loss, train_acc))
            if np.isnan(loss.cpu().item()):
                print('NaN in loss occurred. Aborting training.')
                return -1
            duration += time.time() - before_op_time


            if global_step and global_step % args.log_freq == 0:
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (total_steps / global_step - 1.0) * time_sofar

                print_string = ' train_acc: {:.2f} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                if local_rank == 0:
                    print(print_string.format(train_acc, examples_per_sec, loss, time_sofar, training_time_left))

                writer.add_scalar('train_loss', loss, global_step)
                writer.add_scalar('train_acc', train_acc, global_step)
                writer.add_scalar('learning_rate', current_lr, global_step)
                writer.add_scalar('weight_decay', args.weight_decay, global_step)
                writer.flush()

            if global_step and args.do_online_eval and global_step % args.eval_freq == 0:
                time.sleep(0.1)
                test_acc,test_loss,confusion = test_accuracy(model, test_loader, device=device, local_rank=local_rank)

                writer.add_scalar('test_loss', test_loss, global_step)
                writer.add_scalar('test_acc', test_acc, global_step)

                if test_acc > best_test_acc:
                    patience = 0
                    old_best_acc = best_test_acc
                    old_best_step = best_test_step
                    old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, 'test_acc', old_best_acc)
                    old_model_path = args.log_directory + '/' + args.log_name +'/model/' + old_best_name
                    if os.path.exists(old_model_path):
                        command = 'rm {}'.format(old_model_path)
                        os.system(command)
                    best_test_acc = test_acc
                    best_test_step = global_step
                    model_save_name = '/model-{}-best_{}_{:.5f}'.format(best_test_step, 'test_acc',best_test_acc)
                    if local_rank == 0:
                        print('New best for {}. Saving model: {}'.format('test_acc', model_save_name))
                    # if best_test_acc >= 70:
                    confusion.plot()
                    model_save_path = os.path.join(args.log_directory, args.log_name, 'model')
                    if not os.path.exists(model_save_path):
                        command = 'mkdir ' + model_save_path
                        os.system(command)
                    #torch.save(model.cpu(), model_save_path + model_save_name)
                    torch.save(model.state_dict(), model_save_path + model_save_name)
                        #model.to(device)
                else:
                    patience += 1
            
            if patience > args.patience:
                scheduler.update_lr()
                patience = 0

            model.train()
            global_step += 1
    if local_rank == 0:
        print("finished")

def gather_tensors(target):
    """
    target should be a tensor
    """
    target_list = [torch.ones_like(target) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(target_list, target)
    ret = torch.hstack(target_list)
    
    return ret

def train_accuracy(model, train_loader,criterion_val,device):
    model.eval()
    test_loss = 0.0
    test_total = 0
    correct = 0
    for _, data in enumerate(train_loader):
        # inputs, labels = data
        # inputs, labels = inputs.to(device), labels.to(device)
        # outputs = model(inputs)
        inputs,label = data
        labels = torch.as_tensor(label)
        #inputs, labels = inputs.to(device), labels.to(device)
        inputs,labels = inputs.to(device),labels.to(device)
        # if mixup_fn is not None:
        #     inputs, labels = mixup_fn(inputs, labels)
        # outputs = model(inputs)
        outputs = model(inputs)

        loss = criterion_val(outputs, labels)
        test_loss += loss.item()
        test_total += labels.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().to(device)

    return 100 * correct / test_total,test_loss / len(train_loader)

def test_accuracy(model,test_loader,device,local_rank):
    model.eval()      
    labels_all=[]
    preds_all=[]
    for _, data in enumerate(test_loader):
        # inputs, labels = data
        inputs,label = data
        labels = torch.as_tensor(label)
        inputs,labels = inputs.to(device),labels.to(device)
        # inputs, labels = inputs.to(device), labels.to(device)

        labels = gather_tensors(labels)

        outputs = model(inputs)

        pred = outputs.argmax(dim=1)
        pred = gather_tensors(pred)
        
        labels_np = labels.cpu().numpy()
        labels = labels_np.tolist()
        labels_all.extend(labels)
        preds_cpu = pred.cpu()
        preds_np = preds_cpu.numpy()
        preds = preds_np.tolist()
        preds_all.extend(preds)
    class_indict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4'}
    # 标签名字列表
    label = [label for _, label in class_indict.items()]

    confusion = ConfusionMatrix(args,label)
    conf_matrix,mse=confusion.summary(labels_all,preds_all)

    acc = 1.0*np.trace(conf_matrix)/np.sum(conf_matrix)

    if local_rank == 0:
        print("In test: confusion matrix is:\n {}".format(conf_matrix))
        print('True/Total: {}/{}'.format(np.trace(conf_matrix), np.sum(conf_matrix)))
        print('Acc: {:.3f} ABE: {:.3f}'.format(acc, mse))
    return acc*100,mse,confusion
if __name__ == '__main__':
    train_model(args)