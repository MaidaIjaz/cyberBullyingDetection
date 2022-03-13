import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import nltk.stem
import matplotlib.pyplot as plt
class CyberBullyingDetection:

    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.df_x = None
        self.df_y = None
        self.df = None
        self.cv1 = None
        self.x_traincv = None
        self.x_testcv = None
        self.classifier = None

    ''' ------------------------------------------------------------------------------------------------------------ '''

    def ReadData(self):
        #self.df = pd.read_csv('new_data.csv')
        self.df = pd.read_csv('new_datawithoriginal1.csv')
        #self.df = pd.read_csv('formspring_data_label_new1.csv')
        self.df_x = self.df["post"]
        self.df_y = self.df["class_1"]
        #self.df_x = self.df["Tweet"]
        #self.df_y = self.df["Text Label"]
        #self.df_x = self.df["tweet"]
        #self.df_y = self.df["label"]

    ''' ------------------------------------------------------------------------------------------------------------ '''

    def SplitDataToTrainAndTest(self):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.df_x, self.df_y, test_size=0.3, random_state=1)

    ''' ------------------------------------------------------------------------------------------------------------ '''

    def ReadAndSplitData(self):
        self.ReadData()
        self.SplitDataToTrainAndTest()

    ''' ------------------------------------------------------------------------------------------------------------ '''

    def CountCyberBullyInstancesInWholeData(self):
        print('No. of posts without cyber bullying: ', len(self.df[self.df.class_1 == 0]))
        print('No. of posts with cyber bullying: ', len(self.df[self.df.class_1 == 1]))
        print('\n')

    ''' ------------------------------------------------------------------------------------------------------------ '''

    def TfidVectorizeData(self, analyzer='word', binary=False, decode_error='strict',
                          encoding='utf-8', input='content', lowercase=True, max_df=1.0, max_features=None,
                          min_df=1, ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True, stop_words=None,
                          strip_accents=None,
                          sublinear_tf=True, token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=None, use_idf=True,
                          vocabulary=None):

        self.cv1 = TfidfVectorizer(input=input, encoding=encoding, decode_error=decode_error,
                                   strip_accents=strip_accents,
                                   lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer,
                                   analyzer=analyzer, stop_words=stop_words,
                                   token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                                   max_features=max_features,
                                   vocabulary=vocabulary, binary=binary, norm=norm, use_idf=use_idf,
                                   smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

    ''' ------------------------------------------------------------------------------------------------------------ '''

    def CountVectorizeData(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None,
           lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b',
            ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None,
            binary=False):

        self.cv1 = CountVectorizer(input=input, encoding=encoding, decode_error=decode_error,
                                   strip_accents=strip_accents,
                                   lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer,
                                   analyzer=analyzer, stop_words=stop_words,
                                   token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                                   max_features=max_features,
                                   vocabulary=vocabulary, binary=binary)

    ''' ------------------------------------------------------------------------------------------------------------ '''

    def HashVectorizeData(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None,
           lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b',
            ngram_range=(1, 1), analyzer='word',
            binary=False,n_features = 1048576, norm ='l2', alternate_sign = True, non_negative = False):

        self.cv1 = HashingVectorizer(input=input, encoding=encoding, decode_error=decode_error,
                                   strip_accents=strip_accents,
                                   lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer,
                                   analyzer=analyzer, stop_words=stop_words,
                                   token_pattern=token_pattern, ngram_range=ngram_range, n_features=n_features,
                                   norm=norm,alternate_sign=alternate_sign, non_negative=non_negative, binary=binary)
    def T(self, n_components=2, algorithm='randomized', n_iter=5, random_state=None, tol=0.0):

        self.cv1 = TruncatedSVD(n_components=n_components, algorithm=algorithm, n_iter=n_iter, random_state=random_state,
                                tol=tol)
    ''' ------------------------------------------------------------------------------------------------------------ '''

    def TransformData(self):
        self.x_traincv = self.cv1.fit_transform(self.x_train)
        self.x_testcv = self.cv1.transform(self.x_test)

#       df = pd.DataFrame(self.x_traincv.toarray(), columns=feature_names)
#        df.to_csv('dataset.csv')
#        doc = 1
#        feature_index = self.x_traincv[doc, :].nonzero()[1]
#        tfidf_scores = zip(feature_index, [self.x_traincv[doc, x] for x in feature_index])
#        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
#            print w, s
#        df = pd.DataFrame(data=self.x_traincv.toarray())
#        df.to_csv('outfile.csv', sep=' ', header=False, float_format='%.2f', index=False)
    ''' ------------------------------------------------------------------------------------------------------------ '''

    def PreProcessingMaxAbsScaler(self):
        MAscaler = MaxAbsScaler()
        temp_xTraincv = MAscaler.fit(self.x_traincv).transform(self.x_traincv)
        self.x_traincv = temp_xTraincv
        temp_xTestcv = MAscaler.fit(self.x_testcv).transform(self.x_testcv)
        self.x_testcv = temp_xTestcv

    ''' ------------------------------------------------------------------------------------------------------------ '''

    def PreProcessingStandardScaler(self):
        scaler = StandardScaler(with_mean=False)
        temp_xTraincv = scaler.fit(self.x_traincv).transform(self.x_traincv)
        self.x_traincv = temp_xTraincv
        temp_xTestcv = scaler.fit(self.x_testcv).transform(self.x_testcv)
        self.x_testcv = temp_xTestcv

    ''' ------------------------------------------------------------------------------------------------------------ '''

    def PreProcessingBinarizeData(self):
        binarz = Binarizer()
        temp_xTraincv = binarz.fit(self.x_traincv).transform(self.x_traincv)
        self.x_traincv = temp_xTraincv
        temp_xTestcv = binarz.fit(self.x_testcv).transform(self.x_testcv)
        self.x_testcv = temp_xTestcv

    ''' ------------------------------------------------------------------------------------------------------------ '''

    def PreProcessingNormalizeData(self):
        normaz = Normalizer()
        temp_xTraincv = normaz.fit(self.x_traincv).transform(self.x_traincv)
        self.x_traincv = temp_xTraincv
        temp_xTestcv = normaz.fit(self.x_testcv).transform(self.x_testcv)
        self.x_testcv = temp_xTestcv

    '''' ----------------------------------------------------------------------------------------------------------- '''

    def DecisionTrees(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                      min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False):

        self.classifier = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                 max_features=max_features,
                                                 random_state=random_state,
                                                 max_leaf_nodes=max_leaf_nodes,
                                                 min_impurity_decrease=min_impurity_decrease,
                                                 min_impurity_split=min_impurity_split, class_weight=class_weight,
                                                 presort=presort)

    '''' ----------------------------------------------------------------------------------------------------------- '''

    def KNearestNeighbors(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                          metric='minkowski',
                          metric_params=None, n_jobs=1):

        self.classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                                         leaf_size=leaf_size,
                                                         p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)

    '''' ----------------------------------------------------------------------------------------------------------- '''

    def NeuralNetworks(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001,
                       batch_size='auto',
                       learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                       random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                       nesterovs_momentum=True,
                       early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08):

        self.classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                        alpha=alpha,
                                        batch_size=batch_size, learning_rate=learning_rate,
                                        learning_rate_init=learning_rate_init, power_t=power_t,
                                        max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol,
                                        verbose=verbose, warm_start=warm_start,
                                        momentum=momentum, nesterovs_momentum=nesterovs_momentum,
                                        early_stopping=early_stopping, validation_fraction=validation_fraction,
                                        beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    ''''' ----------------------------------------------------------------------------------------------------------- '''

    def RandomForest(self, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                     min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                     min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                     warm_start=False, class_weight=None):

        self.classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                 max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                                 min_impurity_decrease=min_impurity_decrease,
                                                 min_impurity_split=min_impurity_split, bootstrap=bootstrap,
                                                 oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                                                 verbose=verbose, warm_start=warm_start,
                                                 class_weight=class_weight)

    '''' ----------------------------------------------------------------------------------------------------------- '''

    def MultinomialNaiveBased(self, alpha=0.1, fit_prior=True, class_prior=None):
        self.classifier = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)

    '''' ----------------------------------------------------------------------------------------------------------- '''

    def SupportVectorMachine(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,
                             probability=False,
                             tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                             decision_function_shape='ovr', random_state=None):

        self.classifier = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                              probability=probability,
                              tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose,
                              max_iter=max_iter,
                              decision_function_shape=decision_function_shape, random_state=random_state)

    ''' ------------------------------------------------------------------------------------------------------------ '''

    def TrainAndTestData(self):
        print(self.classifier.fit(self.x_traincv, self.y_train))
        accuracy = self.classifier.score(self.x_testcv, self.y_test)
        print('\nAccuracy on test data is ', accuracy)

    def benchmark(self, clf):
        #score2=0
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(self.x_traincv, self.y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)
        #score1 = 0
        t0 = time()
        pred = clf.predict(self.x_testcv)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(self.y_test, pred)
        print("accuracy:   %0.3f" % score)

        score1 = metrics.precision_score(self.y_test, pred)
        print("precision:   %0.3f" % score1)

        score2 = metrics.recall_score(self.y_test, pred)
        print("recall:   %0.3f" % score2)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])


        print("confusion matrix:")
        print(metrics.confusion_matrix(self.y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, score1, score2


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
'''' ------------------------------------------------------------------------------------------------------------------ '''
if __name__ == "__main__":
    results = []
    print('\n')
    english_stemmer = nltk.stem.SnowballStemmer('english')
    c1 = CyberBullyingDetection()
    c1.ReadAndSplitData()
    c1.CountCyberBullyInstancesInWholeData()
    #english_stemmer = nltk.stem.SnowballStemmer('english')
    # Apply any vectorizer here (Tfid,Count,Hash)
    #c1.CountVectorizeData(min_df =5 ,stop_words ='english',binary=True)
    c1.TfidVectorizeData(max_df=0.5, stop_words='english', binary=True, max_features=2000)

#    yo = StemmedTfidfVectorizer(analyzer='word', binary=True, decode_error='strict',
#                          encoding='utf-8', input='content', lowercase=True, max_df=0.5, max_features=None,
#                          min_df=1, ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True, stop_words='english',
#                          strip_accents=None,
#                          sublinear_tf=True, token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=None, use_idf=True,
#                          vocabulary=None)
    c1.TransformData()

    #  Apply pre processing technique (Binarizer,Standard Scalar,Min Max Scalar,Normalizer,Max Abs Scalar, Robust Scalar)
    #c1.PreProcessingMaxAbsScaler()
    #c1.PreProcessingBinarizeData()
    #c1.PreProcessingNormalizeData()
    #c1.PreProcessingStandardScaler()

    # Apply any classification algorithm and specify the attributes in the bracket
    # DecisionTrees, KNearestNeighbors, NeuralNetworks, RandomForest, Multinomial NaiveBased, SupportVectorMachine

####NeuralNetwork####
    #results.append(c1.benchmark(MLPClassifier(activation='tanh', hidden_layer_sizes=3)))
#    c1.NeuralNetworks(hidden_layer_sizes=3)
#    c1.TrainAndTestData()
    #
    # c1.NeuralNetworks(hidden_layer_sizes=4)
    # c1.TrainAndTestData()
    #
    # c1.NeuralNetworks(hidden_layer_sizes=5)
    # c1.TrainAndTestData()
    #
    # c1.NeuralNetworks(hidden_layer_sizes=6)
    # c1.TrainAndTestData()
    #
    # c1.NeuralNetworks(activation='tanh', hidden_layer_sizes=3)
    # c1.TrainAndTestData()
    #
    #c1.NeuralNetworks(activation='logistic', hidden_layer_sizes=3)
    #c1.TrainAndTestData()
    #
    # c1.NeuralNetworks(activation='identity', hidden_layer_sizes=3)
    # c1.TrainAndTestData()
    #
####KDTree####

    #c1.DecisionTrees(max_depth=21)
    #c1.TrainAndTestData()
    results.append(c1.benchmark(DecisionTreeClassifier(max_depth=7, criterion='gini', splitter='best', min_samples_split=2, min_samples_leaf=1,
                      min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)))

  #  c1.DecisionTrees(splitter='random',max_features='log2')
   # c1.TrainAndTestData()

    #c1.DecisionTrees(splitter='random')
    #c1.TrainAndTestData()

    #c1.DecisionTrees(criterion='entropy')
    #c1.TrainAndTestData()

####NaiveBayes####
    results.append(c1.benchmark(MultinomialNB(alpha=.01)))
    #c1.MultinomialNaiveBased()
    #c1.TrainAndTestData()

####KNN####
    results.append(c1.benchmark(KNeighborsClassifier(n_neighbors=5)))
    # c1.KNearestNeighbors(n_neighbors=1,weights ='uniform', p=1)
    # c1.TrainAndTestData()
    #
    # c1.KNearestNeighbors(n_neighbors=4)
    # c1.TrainAndTestData()
    #
    # c1.KNearestNeighbors(n_neighbors=5,weights ='uniform', p=1)
    # c1.TrainAndTestData()
    #
    # c1.KNearestNeighbors(n_neighbors=8)
    # c1.TrainAndTestData()
    #
    # c1.KNearestNeighbors(n_neighbors=10,weights ='uniform', p=1)
    # c1.TrainAndTestData()
    #
    # c1.KNearestNeighbors(n_neighbors=12)
    # c1.TrainAndTestData()

####SVC####
    #results.append(c1.benchmark(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)))
    # c1.SupportVectorMachine(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    # c1.TrainAndTestData()
    #
    # c1.SupportVectorMachine(C=2.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    # c1.TrainAndTestData()
    #
    # c1.SupportVectorMachine(C=4.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    # c1.TrainAndTestData()
    #
    # c1.SupportVectorMachine(C=5.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    # c1.TrainAndTestData()
    #
####RandomForestClassifier####
    results.append(c1.benchmark(RandomForestClassifier(n_estimators=5, max_features='sqrt')))
    #c1.RandomForest(n_estimators=40, max_features='sqrt')
    #c1.TrainAndTestData()
    #
    # c1.RandomForest(n_estimators=17, max_features='sqrt')
    # c1.TrainAndTestData()
    #
    # c1.RandomForest(n_estimators=15, max_features='log2')
    # c1.TrainAndTestData()
    #
    # c1.RandomForest(n_estimators=17, max_features='log2')
    # c1.TrainAndTestData()
    #
    # c1.RandomForest(n_estimators=15, max_features='sqrt', min_impurity_split=.1)
    # c1.TrainAndTestData()
    #
    # c1.RandomForest(n_estimators=15, max_features='sqrt',min_weight_fraction_leaf=0.1)
    # c1.TrainAndTestData()
    #
#    c1.RandomForest(n_estimators=15, max_features='sqrt',min_impurity_split=.1,min_weight_fraction_leaf=0.1)
    # c1.TrainAndTestData()
    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, score1, score2 = results

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="accuracy", color='navy')
    plt.barh(indices + .3, score1, .2, label="precision",
             color='c')
    plt.barh(indices + .6, score2, .2, label="recall", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()


