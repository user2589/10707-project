

# 10707 project

Most of files in this directory were cloned from BERT repository:
    https://github.com/google-research/bert
    
# How to run:

    # replace names with those which actually have embeddings
    ./name_sub.py -i input/test_stage_1.tsv -o input/test_stage_1_name_sub.csv 
    # get bert contextual embeddings
    ./get_bert_embeddings.py -i input/test_stage_1_name_sub.csv -o input/test_stage_1_bert_embed.csv