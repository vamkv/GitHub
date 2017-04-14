import sframe
import sklearn.tree,numpy

loans = sframe.SFrame('lending-club-data.gl/')
print loans.column_names()

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')


print ' percentage of loans that are safe: %f ' %( ((loans['safe_loans']==1).sum()/float(len(loans)))*100 )
print ' percentage of loans that are risky: %f ' %( ((loans['safe_loans']==-1).sum()/float(len(loans)))*100 )

# Features we are going to use 
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)

#For scikit-learn's decision tree implementation, it requires numerical values for it's data matrix. 
#This means you will have to turn categorical variables into binary features via one-hot encoding. 

categorical_variables = []
for feat_name, feat_type in zip(loans_data.column_names(), loans_data.column_types()):
    if feat_type == str:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)

train_data, validation_data = loans_data.random_split(.8, seed=1)

#You will have to first convert the SFrame into a numpy data matrix, 
#and extract the target labels as a numpy array (Hint: you can use the .to_numpy()
train_data_output=train_data['safe_loans']
train_data_input=train_data.remove_column('safe_loans')

X= train_data_input.to_numpy()
y=train_data_output.to_numpy()

clf=sklearn.tree.DecisionTreeClassifier(max_depth=6)
decision_tree_model=clf.fit(X,y)

clf2=sklearn.tree.DecisionTreeClassifier(max_depth=2)
small_model=clf2.fit(X,y)

clf3=sklearn.tree.DecisionTreeClassifier(max_depth=10)
big_model=clf3.fit(X,y)

# Making predictions
# Let's consider two positive and two negative examples from the validation set and see what 
#the model predicts. We will do the following:
#Predict whether or not a loan is safe.
#Predict the probability that a loan is safe.
#First, let's grab 2 positive examples and 2 negative examples. In SFrame, that would be:

validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]
sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]
sample_validation_data = (sample_validation_data_safe.append(sample_validation_data_risky))
sample_validation_data_classes=sample_validation_data['safe_loans']
sample_y=sample_validation_data_classes.to_numpy()

# removing the label/class from the data
sample_X=(sample_validation_data.remove_column('safe_loans')).to_numpy()

print 'actual classes for sample data are: %r' %(sample_validation_data_classes) 
print 'predicted classes from decision_tree_model are:%r ' %(decision_tree_model.predict(sample_X))
print 'predicted probabilities for classes from decision_tree_model are:%r ' %(decision_tree_model.predict_proba(sample_X))
print 'predicted probabilities for classes from small_model are:%r ' %(small_model.predict_proba(sample_X))
print 'scores for decision_tree_model are:%r ' %(decision_tree_model.score(sample_X,sample_y))
print 'scores for small_model are:%r ' %(small_model.score(sample_X,sample_y))

validtn_data=validation_data[:]
# Predictions on actual validation set
validation_data_y=validation_data['safe_loans'].to_numpy()
validation_data_X=validation_data.remove_column('safe_loans').to_numpy()

print 'accuracy of total validatin set using decision_tree_model is:%r' %(clf.score(validation_data_X,validation_data_y))
print 'accuracy of total validatin set using small_model is:%r' %(clf2.score(validation_data_X,validation_data_y))
print 'accuracy of total validatin set using big_model is:%r' %(clf3.score(validation_data_X,validation_data_y))

# Total erros in decision tree model
validation_data_pos_class= validtn_data[validtn_data['safe_loans'] == 1]
validation_data_neg_class= validtn_data[validtn_data['safe_loans'] == -1]

# Total Fasle Negatives
FN= ( validation_data_pos_class['safe_loans'].to_numpy()!= decision_tree_model.predict(validation_data_pos_class.remove_column('safe_loans').to_numpy()) ).sum()

# Total Fasle Positives
FP= ( validation_data_neg_class['safe_loans'].to_numpy()!= decision_tree_model.predict(validation_data_neg_class.remove_column('safe_loans').to_numpy()) ).sum()

print 'Total loss is %f' %( (FN*10000)+(FP*20000) )