import os
import pandas as pd 

def rename_cols_of_the_table(original_table , data_root : str, item_dictionary_name : str ='d_items.csv'):
    item_dictionary_path=os.path.join(data_root,item_dictionary_name)
    item_dictionary=pd.read_csv(item_dictionary_path)
    item_dictionary.itemid=item_dictionary.itemid.astype(str)
    code_to_name=dict(zip(item_dictionary.itemid, item_dictionary.abbreviation))
    code_to_name_keys=code_to_name.keys()
    renamed_table=original_table.copy()
    original_columns=original_table.columns
    for orig_col in original_columns:
        code=orig_col[1]
        if code in code_to_name_keys:
            name=(orig_col[0],code_to_name[code])
            renamed_table=renamed_table.rename(columns={orig_col: name})
        # else: #leaving out the ICD-10 codes
        #     renamed_table=renamed_table.drop(columns=[orig_col])

    return renamed_table










