
from utils.utils import *
from model.FADFD import FADFD
from data_factory.data_loader import get_loader_segment
from metrics.metrics import *
import warnings
import pandas as pd
# import torch.nn as nn
warnings.filterwarnings('ignore')


def get_events(y_test, outlier=1, normal=0):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
        else:
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events


def get_composite_fscore_raw(pred_labels, true_events, y_test, return_prec_rec=False):
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp / (tp + fn)
    prec_t = precision_score(y_test, pred_labels)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c

        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size,win_size_1=self.win_size_1,count=self.count, mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size,win_size_1=self.win_size_1,count=self.count, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size,win_size_1=self.win_size_1,count=self.count, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size,win_size_1=self.win_size_1,count=self.count, mode='test', dataset=self.dataset)

        self.build_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    def build_model(self):
        self.model = FADFD(p=self.p,select=self.select)
        
        if torch.cuda.is_available():
            self.model.cuda()


    def test(self):
        # find the threshold
        attens_energy = []
        for i, (input_data, data_global, labels) in enumerate(self.thre_loader):

            input = input_data.float().to(self.device)  # (128,100,51)
            data_global = data_global.float().to(self.device)
            score = self.model(input, data_global)

            metric = score.unsqueeze(-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0)
        test_energy = np.array(attens_energy)

        thresh = np.percentile(test_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        test_labels = []
        attens_energy = []
        for i, (input_data, data_global, labels) in enumerate(self.thre_loader):

            input = input_data.float().to(self.device)  # (128,100,51)
            data_global = data_global.float().to(self.device)

            score = self.model(input, data_global)

            metric = score.unsqueeze(-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)


        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        #############################################################################################################
        from metrics.basic_metrics import basic_metricor, generate_curve
        pred = pred.reshape(-1)
        test_energy = test_energy.reshape(-1)
        grader = basic_metricor()
        PointF1 = grader.metric_PointF1(gt, test_energy, preds=pred)
        PointF1PA = grader.metric_PointF1PA(gt, test_energy, preds=pred)
        EventF1PA = grader.metric_EventF1PA(gt, test_energy, preds=pred)
        RF1 = grader.metric_RF1(gt, test_energy, preds=pred)
        # Affiliation_F = grader.metric_Affiliation(gt, test_energy, preds=pred)
        print(
            "PointF1 : {:0.4f}, PointF1PA : {:0.4f}, EventF1PA : {:0.4f}, RF1 : {:0.4f} ".format(
                round(PointF1, 4), round(PointF1PA, 4), round(EventF1PA, 4), round(RF1, 4)
            )
        )
        true_events = get_events(gt)
        prec_t, rec_e, fscore_c = get_composite_fscore_raw(pred, true_events, gt, return_prec_rec=True)
        print("F1_c:", round(fscore_c, 4))
        print("prec_t:", round(prec_t, 4))
        print("rec_e:", round(rec_e, 4))

        from metrics.EventKthF1PA import EventKthF1PA

        eval_f1 = EventKthF1PA(20, mode="raw")

        from sklearn.metrics import f1_score

        print("binary", round(f1_score(gt, pred, average='binary'), 4))
        print('\n')
        print("micro:      ", round(f1_score(gt, pred, average='micro'), 4))
        print("macro:      ", round(f1_score(gt, pred, average='macro'), 4))
        print("weighted:   ", round(f1_score(gt, pred, average='weighted'), 4))
        # print("samples", f1_score(gt, pred, average='samples'))
        # print("none", f1_score(gt, pred, average=None))

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        from metrics.combine_all_scores import combine_all_evaluation_scores

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

        matrix = []
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))
        #############################################################################################################

        matrix = [self.index]

        input_show=np.concatenate([np.expand_dims(test_labels,axis=-1),attens_energy],axis=1)
        df = pd.DataFrame(input_show)
        # excel_writer = pd.ExcelWriter('tensor_data.xlsx', engine='openpyxl')  # 选择 'xlsxwriter' 或 'openpyxl' 作为引擎
        # df.to_excel(excel_writer, index=False, sheet_name='Sheet1')
        # excel_writer.save()

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))



        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/'+self.data_path+'.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score



#     def test(self):
#         # 检查 thre_loader 是否为空
#         if len(self.thre_loader) == 0:
#             raise ValueError("thre_loader is empty! Check data loading.")
#
#         # 计算阈值
#         attens_energy = []
#         for i, (input_data, data_global, labels) in enumerate(self.thre_loader):
#             input = input_data.float().to(self.device)  # (128, 100, 51)
#             data_global = data_global.float().to(self.device)
#             score = self.model(input, data_global)
#
#             # 检查模型输出是否为空
#             if score is None:
#                 raise ValueError("Model output is None! Check model implementation.")
#
#             metric = score.unsqueeze(-1)
#             cri = metric.detach().cpu().numpy()
#             attens_energy.append(cri)
#
#         # 检查 attens_energy 是否为空
#         if len(attens_energy) > 0:
#             attens_energy = np.concatenate(attens_energy, axis=0)
#         else:
#             raise ValueError("attens_energy is empty! Check data generation.")
#
#         test_energy = np.array(attens_energy)
#         thresh = np.percentile(test_energy, 100 - self.anormly_ratio)
#
#         # 计算预测结果
#         test_labels = []
#         attens_energy = []
#         for i, (input_data, data_global, labels) in enumerate(self.thre_loader):
#             input = input_data.float().to(self.device)  # (128, 100, 51)
#             data_global = data_global.float().to(self.device)
#             score = self.model(input, data_global)
#
#             metric = score.unsqueeze(-1)
#             cri = metric.detach().cpu().numpy()
#             attens_energy.append(cri)
#             test_labels.append(labels)
#
#         # 检查 attens_energy 和 test_labels 是否为空
#         if len(attens_energy) > 0 and len(test_labels) > 0:
#             attens_energy = np.concatenate(attens_energy, axis=0)
#             test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
#         else:
#             raise ValueError("attens_energy or test_labels is empty! Check data generation.")
#
#         test_energy = np.array(attens_energy)
#         test_labels = np.array(test_labels)
#
#         # 计算预测结果
#         pred = (test_energy > thresh).astype(int)
#         gt = test_labels.astype(int)
#
#         # 处理异常状态
#         anomaly_state = False
#         for i in range(len(gt)):
#             if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
#                 anomaly_state = True
#                 for j in range(i, 0, -1):
#                     if gt[j] == 0:
#                         break
#                     else:
#                         if pred[j] == 0:
#                             pred[j] = 1
#                 for j in range(i, len(gt)):
#                     if gt[j] == 0:
#                         break
#                     else:
#                         if pred[j] == 0:
#                             pred[j] = 1
#             elif gt[i] == 0:
#                 anomaly_state = False
#             if anomaly_state:
#                 pred[i] = 1
#
#         # 计算性能指标
#         from sklearn.metrics import precision_recall_fscore_support, accuracy_score
#         accuracy = accuracy_score(gt, pred)
#         precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
#         print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
#             accuracy, precision, recall, f_score))
#
#         # 保存结果
#         if self.data_path == 'UCR' or self.data_path == 'UCR_AUG':
#             import csv
#             with open('result/' + self.data_path + '.csv', 'a+') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([self.index, accuracy, precision, recall, f_score])  # 保存指标
#
#         return accuracy, precision, recall, f_score
