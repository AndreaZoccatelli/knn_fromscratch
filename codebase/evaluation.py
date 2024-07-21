import numpy as np


class EvaluationTools:
    def __init__(self, y_pred: np.array, y_true: np.array) -> None:
        self.y_pred = y_pred
        self.y_true = y_true

    def accuracy_score(self):
        self.accuracy_score = np.mean(self.y_pred == self.y_true)
        print("Accuracy score:\n", self.accuracy_score, "\n")

    def f1_score(self):
        classes = np.unique(self.y_true)
        n_classes = len(classes)
        self.f1_score_dict = {}
        f1_score_cum = 0

        for i in classes:
            class_i_pred_pos = np.where(self.y_pred == 0)[0]
            class_i_real_pos = np.where(self.y_true == 0)
            TP_i = len(np.intersect1d(class_i_pred_pos, class_i_real_pos))
            FP_i = len(np.setdiff1d(class_i_pred_pos, class_i_real_pos))
            FN_i = len(np.setdiff1d(class_i_real_pos, class_i_pred_pos))
            Precision = TP_i / (TP_i + FP_i)
            Recall = TP_i / (TP_i + FN_i)

            f1_score_i = (2 * Precision * Recall) / (Precision + Recall)
            f1_score_cum += f1_score_i
            self.f1_score_dict[i] = f1_score_i

        self.f1_score_avg = f1_score_cum / n_classes
        print("f1 Scores by classes:")
        print(self.f1_score_dict, "\n")
        print("Avg. f1 score:")
        print(self.f1_score_avg, "\n")
