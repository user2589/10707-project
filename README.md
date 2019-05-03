

# 10707 project

Most of files in this directory were cloned from BERT repository:
    https://github.com/google-research/bert
    
    
# How to run, end to end:

    # replace names with those which actually have embeddings
    ./name_sub.py -i input/test_stage_1.tsv -o input/test_stage_1_name_sub.csv 
    ./name_sub.py -i input/test_stage_2.tsv -o input/test_stage_2_name_sub.csv 

    # get bert contextual embeddings - takes couple hours
    ./get_bert_embeddings.py -i input/test_stage_1_name_sub.csv -o input/test_stage_1_bert_embed.csv
    ./get_bert_embeddings.py -i input/test_stage_2_name_sub.csv -o input/test_stage_2_bert_embed.csv


    # train neural network (run in Python3):    
    be1 = pd.read_csv('input/test_stage_1_bert_embed.csv', index_col=0).applymap(                                            
        lambda value: np.array([float(v) for v in value.split(",")])) 
    td1 = np.array(list(be1.apply(                                                                                           
        lambda row: np.concatenate(row[['Pronoun', 'A', 'B']]), axis=1)))
    be2 = pd.read_csv('input/test_stage_2_bert_embed.csv', index_col=0).applymap(                                             
         lambda value: np.array([float(v) for v in value.split(",")]))
    td2 = np.array(list(be2.apply(                                                                                             
        lambda row: np.concatenate(row[['Pronoun', 'A', 'B']]), axis=1)))
    embeddings_df = pd.read_csv('input/test_stage_2_name_sub.csv', index_col=0)
     
    model = tf.keras.Sequential([
        layers.Dropout(0.7),
        layers.Dense(1024, activation='sigmoid', input_shape=(2304,)),
        layers.Dropout(0.6),
        layers.Dense(256, activation='sigmoid'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='sigmoid'),
        layers.Dropout(0.4),
        layers.Dense(3, activation='softmax')
    ])
     model.compile(
         optimizer=tf.train.AdamOptimizer(0.001),
         loss = 'categorical_crossentropy',
          metrics = ['accuracy'])
     model.fit(td1, labels,
         validation_split=0.1,  # comment out for the final model 
         batch_size=5, epochs=120)

     pd.DataFrame(
         model.predict(td2),
         columns=['A', 'B', 'NEITHER'],
         index=embeddings_df.index).to_csv('input/test_stage_2_predictions.csv')


Submit the resulting on Kaggle:
