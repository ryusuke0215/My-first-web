import numpy as np

class Node(object):
    def __init__(self, criterion = "gini", max_depth = None, random_state = None):
        self.criterion    = criterion
        self.max_depth    = max_depth
        self.random_state = random_state
        self.depth        = None
        self.left         = None
        self.right        = None
        self.feature      = None
        self.threshold    = None # threshold : 闘値
        self.label        = None
        self.impurity     = None # impurity : 不純物
        self.info_gain    = None
        self.num_samples  = None
        self.num_classes  = None
        
    def split_node(self, sample, target, depth, ini_num_classes):
        # sample : ノード内の訓練データ
        # target : ノード内の正解ラベル。１次元
        # depth : 木の深さ
        # ini_num_classes : 初期のクラス
        
        self.depth       = depth
        self.num_samples = len(target) # データの個数
        
        # ノードの中にある各クラスの個数
        self.num_classes = [len(target[target == i]) for i in ini_num_classes]
        
        # ノード内に1つのクラスしかない場合
        if len(np.unique(target)) == 1:
            self.label = target[0]
            self.impurity = self.criterion_func(target)
            return
        
        # ノード内にあるクラス名とクラス数を対応づけた辞書を作る
        class_count = {i: len(target[target == i])for i in np.unique(target)}
        # 辞書の中身を取り出し、個数が一番多いクラス番号を取ってくる
        # items(): 辞書内のkeyとvalueを持ってくる
        # max(, key): keyに関数を渡すとその処理をしたあとのmaxが帰ってくる
        # (key, value)が帰ってくるので[0]でkeyを取ってくる
        self.label = max(class_count.items(), key = lambda x : x[1])[0]
        
        #if depth == self.max_depth:
        #    return
        
        # targetの不純度
        self.impurity = self.criterion_func(target)
        
        # 特徴の個数
        num_features = sample.shape[1]
        self.info_gain = 0.0
        
        if self.random_state != None:
            np.random.seed(self.random_state)
            
        # 0 ~ num_features - 1 にある整数を num_features 個ランダムに選ぶ
        f_loop_order = np.random.permutation(num_features).tolist()
        
        for f in f_loop_order:
            # 全体からf番目のデータを持ってくる重複はゆるさない
            uniq_feature = np.unique(sample[:, f])
            # uniq_feature[:-1] : 0番目から最後から2番目まで
            # uniq_fature[1:] : 1番目から最後まで
            split_points = (uniq_feature[:-1] + uniq_feature[1:]) / 2.0
            
            # 一番うまく分類できるthresholdの値(閾値)を探す
            for threshold in split_points:
                # sample全体のf番目がthreshold以下だったその要素番号のtargetの値をtarget_lにリスト型で格納
                target_l = target[sample[:, f] <= threshold]
                #sample全体のf番目がthresholdより大きかったらその要素番号のtargetの値をtarget_rにリスト型で格納
                target_r = target[sample[:, f] > threshold]
                # 情報利得を得る
                val = self.calc_info_gain(target, target_l, target_r)
                # 情報利得が最大の情報利得を上回ったとき
                if self.info_gain < val:
                    self.info_gain = val# 情報利得を更新
                    self.feature = f# 特徴をfにする
                    self.threshold = threshold# thresholdの値を更新
                    
        if self.info_gain == 0.0:
            return
        
        if depth == self.max_depth:
            return
        
        # 全体のデータからself.threshold以下のself.featureを格納する
        sample_l = sample[sample[:, self.feature] <= self.threshold]
        target_l = target[sample[:, self.feature] <= self.threshold]
        # 再帰呼び出しで木を左側にさらに深くする
        self.left = Node(self.criterion, self.max_depth)
        self.left.split_node(sample_l, target_l, depth + 1, ini_num_classes)
        # 全体のデータからself.thresholdより大きいself.featureを格納する
        sample_r = sample[sample[:, self.feature] > self.threshold]
        target_r = target[sample[:, self.feature] > self.threshold]
        # 再帰呼び出しで木を右側にさらに深くする
        self.right = Node(self.criterion, self.max_depth)
        self.right.split_node(sample_r, target_r, depth + 1, ini_num_classes)
        
    def criterion_func(self, target):
        classes = np.unique(target)
        numdata = len(target)
        
        # ΣiΣj | xi - xj | / (2 * n^2 * x) 
        # X = [0, 1, 0, 2, 0, ...]
        # 1 - Σi(count(X==xi) / numdata)^2
        if self.criterion == "gini":
            val = 1
            for c in classes:
                p = float(len(target[target == c])) / numdata
                val -= p**2.0
                
        elif self.criterion == "entropy":
            val = 0
            for c in classes:
                p = float(len(target[target == c])) / numdata
                if p != 0.0:
                    val -= p * np.log2(p)
                
        return val
    
    def calc_info_gain(self, target_p, target_cl, target_cr):
        cri_p = self.criterion_func(target_p)# 親ノードの不純度
        cri_cl = self.criterion_func(target_cl)# 子ノード(左)の不純度
        cri_cr = self.criterion_func(target_cr)# 子ノード(右)の不純度
        # 情報利得を返す
        return cri_p - len(target_cl) / float(len(target_p)) * cri_cl - len(target_cr) / float(len(target_p)) * cri_cr
    
    def predict(self, sample):
        if self.feature == None or self.depth == self.max_depth:
            return self.label
        else:
            if sample[self.feature] <= self.threshold:
                return self.left.predict(sample)
            else:
                return self.right.predict(sample)

class DecisionTree(object):
    def __init__(self, criterion = "gini", max_depth = None, random_state = None):
        self.tree = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        
    def fit(self, sample, target):
        self.tree = Node(self.criterion, self.max_depth, self.random_state)
        self.tree.split_node(sample, target, 0, np.unique(target))
        
    def predict(self, sample):
        pred = []
        for s in sample:
            pred.append(self.tree.predict(s))
        return np.array(pred)
    
    def score(self, sample, target):
        return sum(self.predict(sample) == target) / float(len(target))
